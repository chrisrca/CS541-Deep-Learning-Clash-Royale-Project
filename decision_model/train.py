import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from huggingface_hub import list_repo_files, hf_hub_download
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb

from model import ConvLSTMClashRoyaleModel
from dataset import ClashRoyaleDataset, ALL_CARDS

# def get_hf_parquet_local_paths(repo_id: str, repo_type: str = "dataset"):
#     """List all .parquet files in a Hugging Face repo and download them locally.

#     Returns a list of local cached file paths suitable for ParquetClashDataset.
#     """
#     files = list_repo_files(repo_id, repo_type=repo_type)
#     parquet_files = [f for f in files if f.endswith(".parquet")]
#     if not parquet_files:
#         raise ValueError(f"No .parquet files found in HF repo: {repo_id}")

#     local_paths = []
#     for fp in parquet_files:
#         local_path = hf_hub_download(repo_id=repo_id, filename=fp, repo_type=repo_type)
#         local_paths.append(local_path)
#     return local_paths


def build_dataloaders(config, device):
    # hf_repo_id = config["hf_repo_id"]
    # hf_repo_type = config.get("hf_repo_type", "dataset")
    # parquet_paths = get_hf_parquet_local_paths(hf_repo_id, repo_type=hf_repo_type)

    parquet_paths = ["./new_arena_placement.parquet", "./Nones_arena_21.parquet", "./Nones_arena_22.parquet"]
    dataset = ClashRoyaleDataset(parquet_paths, config["grid_w"], config["grid_h"], config["num_cards"])

    val_ratio = config.get("val_ratio", 0.1)
    test_ratio = config.get("test_ratio", 0.1)
    assert 0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0 and val_ratio + test_ratio < 1.0

    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test])

    print(f"Dataset sizes -> total: {n_total}, train: {n_train}, val: {n_val}, test: {n_test}")

    # On Windows, multiprocessing DataLoader workers (num_workers>0) often cause
    # pickling errors like `OSError: [Errno 22] Invalid argument`. To avoid this,
    # we force single-process loading with num_workers=0.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_loader, test_loader, dataset


def compute_action_pos_weight(dataset, device):
    """Compute pos_weight for BCEWithLogitsLoss to balance action classes.
    
    pos_weight = num_negative / num_positive
    This gives higher weight to the minority class.
    """
    num_positive = 0  # action = 1 (play a card)
    num_negative = 0  # action = 0 (no-op)
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        action = sample["label_action"].item()
        if action == 1:
            num_positive += 1
        else:
            num_negative += 1
    
    if num_positive == 0:
        pos_weight = 1.0
    else:
        pos_weight = num_negative / num_positive
    
    print(f"Action class balance: positive={num_positive}, negative={num_negative}, pos_weight={pos_weight:.4f}")
    
    return torch.tensor([pos_weight], device=device)


def build_model(config, device):
    # num_cards is the number of actual cards (no no-op class)
    num_cards = config["num_cards"]
    grid_h = config["grid_h"]
    grid_w = config["grid_w"]
    numeric_feat_dim = config["numeric_feat_dim"]
    convlstm_hidden = config["convlstm_hidden"]
    backbone_proj_channels = config["backbone_proj_channels"]
    use_pretrained = config.get("backbone_pretrained", False)
    pretrained_weights = None
    if use_pretrained:
        from torchvision.models import MobileNet_V2_Weights

        pretrained_weights = MobileNet_V2_Weights.DEFAULT

    model = ConvLSTMClashRoyaleModel(
        num_cards=num_cards,
        numeric_feat_dim=numeric_feat_dim,
        grid_h=grid_h,
        grid_w=grid_w,
        convlstm_hidden=convlstm_hidden,
        pretrained_backbone_weights=pretrained_weights,
        backbone_proj_channels=backbone_proj_channels,
    )
    return model.to(device)


def train_one_epoch(model, train_loader, optimizer, action_loss_fn, card_loss_fn, place_loss_fn, device, config, epoch, rolling_losses=None, scaler=None, use_amp=False):  # type: ignore[type-arg]
    """Train for one epoch with optional mixed precision."""
    assert scaler is not None, "scaler must be provided"
    
    model.train()
    total_loss = 0.0
    total_action_loss = 0.0
    total_card_loss = 0.0
    total_place_loss = 0.0
    total_batches = 0
    total_action_batches = 0  # Batches where we compute card loss (action=1)

    # Store per-batch losses to compute rolling averages for logging.
    # Use persistent dict that carries over across epochs if provided
    if rolling_losses is None:
        rolling_losses = {
            "batch": [],
            "action": [],
            "card": [],
            "place": [],
        }
    
    batch_losses = rolling_losses["batch"]
    batch_action_losses = rolling_losses["action"]
    batch_card_losses = rolling_losses["card"]
    batch_place_losses = rolling_losses["place"]

    frames_per_sample = config["frames_per_sample"]

    num_epochs = config.get("num_epochs", None)
    if num_epochs is not None:
        print(f"Starting training epoch {epoch + 1}/{num_epochs}...")
    else:
        print(f"Starting training epoch {epoch + 1}...")

    for batch_idx, batch in enumerate(train_loader):
        # Use non_blocking=True to overlap CPU->GPU transfer with computation
        frames = batch["frames"].to(device, non_blocking=True)
        numeric_features = batch["numeric_features"].to(device, non_blocking=True)
        labels_action = batch["label_action"].long().to(device, non_blocking=True)
        labels_card = batch["label_card"].long().to(device, non_blocking=True)
        labels_placement = batch["label_placement"].long().to(device, non_blocking=True).view(-1)
        
        playable_mask = batch["playable_mask"].to(device, non_blocking=True)
        hand_mask = batch["hand_mask"].to(device, non_blocking=True)

        if frames.ndim == 4:
            frames = frames.unsqueeze(1)

        T = frames.shape[1]
        use_T = min(frames_per_sample, T)
        frames = frames[:, -use_T:, ...]

        optimizer.zero_grad(set_to_none=True)
        
        # Use automatic mixed precision for forward pass
        with torch.amp.autocast("cuda", enabled=use_amp): # type: ignore
            outputs = model(frames, numeric_features, playable_mask, hand_mask)
            action_logits = outputs["action_logits"]  # (B, 1)
            card_logits = outputs["card_logits"]  # (B, num_cards)
            placement_logits = outputs["placement_logits"]  # (B, num_cards, grid_cells)

            # Action loss: binary classification (no-op vs play)
            action_loss = action_loss_fn(action_logits.squeeze(1), labels_action.float())
            
            # Card loss and placement loss: only for samples where action=1 (play a card)
            action_mask = labels_action == 1
            
            if action_mask.any():
                # Filter to samples where a card was played
                card_logits_masked = card_logits[action_mask]
                labels_card_masked = labels_card[action_mask]
                placement_logits_masked = placement_logits[action_mask]
                labels_placement_masked = labels_placement[action_mask]
                
                # Card loss
                card_loss = card_loss_fn(card_logits_masked, labels_card_masked)
                
                # Placement loss: gather logits for the ground-truth card
                B_masked = placement_logits_masked.shape[0]
                relevant_placement_logits = placement_logits_masked[
                    torch.arange(B_masked, device=device), labels_card_masked, :
                ]
                place_loss = place_loss_fn(relevant_placement_logits, labels_placement_masked)
            else:
                # No samples with action=1 in this batch
                card_loss = torch.tensor(0.0, device=device)
                # Use rolling average for placement loss if available
                if batch_place_losses:
                    window = min(len(batch_place_losses), config.get("rolling_average_window", 100))
                    recent_losses = batch_place_losses[-window:]
                    place_loss = torch.tensor(sum(recent_losses) / len(recent_losses), device=device)
                else:
                    place_loss = torch.tensor(0.0, device=device)
            
            loss = action_loss + card_loss + place_loss

        # Scale loss and backward pass
        scaler.scale(loss).backward()

        max_grad_norm = config.get("max_grad_norm", 0.0)
        if max_grad_norm and max_grad_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_action_loss += action_loss.item()
        if action_mask.any():
            total_card_loss += card_loss.item()
            total_place_loss += place_loss.item()
            total_action_batches += 1
        total_batches += 1

        batch_losses.append(loss.item())
        batch_action_losses.append(action_loss.item())
        if action_mask.any():
            batch_card_losses.append(card_loss.item())
            batch_place_losses.append(place_loss.item())

        # Compute rolling average over the last N batches
        window = config["rolling_average_window"]
        start_idx = max(0, len(batch_losses) - window)
        window_losses = batch_losses[start_idx:]
        window_action_losses = batch_action_losses[start_idx:]
        window_card_losses = batch_card_losses[max(0, len(batch_card_losses) - window):]
        window_place_losses = batch_place_losses[max(0, len(batch_place_losses) - window):]
        
        avg_window_loss = sum(window_losses) / max(len(window_losses), 1)
        avg_window_action = sum(window_action_losses) / max(len(window_action_losses), 1)
        avg_window_card = sum(window_card_losses) / max(len(window_card_losses), 1) if window_card_losses else 0.0
        avg_window_place = sum(window_place_losses) / max(len(window_place_losses), 1) if window_place_losses else 0.0

        wandb.log(
            {
                "train/loss": avg_window_loss,
                "train/action_loss": avg_window_action,
                "train/card_loss": avg_window_card,
                "train/place_loss": avg_window_place,
            }
        )

        # Only print every log_every batches
        if (batch_idx + 1) % config["log_every"] == 0:
            print(
                f"[Train] epoch {epoch + 1}, batch {batch_idx + 1}/{len(train_loader)} "
                f"loss={avg_window_loss:.4f}, action={avg_window_action:.4f}, card={avg_window_card:.4f}, place={avg_window_place:.4f}"
            )

    return rolling_losses


def evaluate(model, data_loader, action_loss_fn, card_loss_fn, place_loss_fn, device, config, epoch, split_name):
    model.eval()
    total_loss = 0.0
    total_action_loss = 0.0
    total_card_loss = 0.0
    total_place_loss = 0.0
    total_batches = 0
    total_action_batches = 0  # Batches with action=1 samples
    
    # Collect predictions and labels for metrics
    all_action_preds = []
    all_action_labels = []
    all_card_preds = []  # Only for samples where action=1
    all_card_labels = []  # Only for samples where action=1

    frames_per_sample = config["frames_per_sample"]

    with torch.no_grad():
        for batch in data_loader:
            # Use non_blocking=True to overlap CPU->GPU transfer with computation
            frames = batch["frames"].to(device, non_blocking=True)
            numeric_features = batch["numeric_features"].to(device, non_blocking=True)
            labels_action = batch["label_action"].long().to(device, non_blocking=True)
            labels_card = batch["label_card"].long().to(device, non_blocking=True)
            labels_placement = batch["label_placement"].long().to(device, non_blocking=True).view(-1)
            
            playable_mask = batch["playable_mask"].to(device, non_blocking=True)
            hand_mask = batch["hand_mask"].to(device, non_blocking=True)

            if frames.ndim == 4:
                frames = frames.unsqueeze(1)

            T = frames.shape[1]
            use_T = min(frames_per_sample, T)
            frames = frames[:, -use_T:, ...]

            outputs = model(frames, numeric_features, playable_mask, hand_mask)
            action_logits = outputs["action_logits"]  # (B, 1)
            card_logits = outputs["card_logits"]  # (B, num_cards)
            placement_logits = outputs["placement_logits"]  # (B, num_cards, grid_cells)

            # Action loss
            action_loss = action_loss_fn(action_logits.squeeze(1), labels_action.float())
            total_action_loss += action_loss.item()
            
            # Card and placement loss: only for samples where action=1
            action_mask = labels_action == 1
            
            if action_mask.any():
                card_logits_masked = card_logits[action_mask]
                labels_card_masked = labels_card[action_mask]
                placement_logits_masked = placement_logits[action_mask]
                labels_placement_masked = labels_placement[action_mask]
                
                card_loss = card_loss_fn(card_logits_masked, labels_card_masked)
                
                B_masked = placement_logits_masked.shape[0]
                relevant_placement_logits = placement_logits_masked[
                    torch.arange(B_masked, device=device), labels_card_masked, :
                ]
                place_loss = place_loss_fn(relevant_placement_logits, labels_placement_masked)
                
                total_card_loss += card_loss.item()
                total_place_loss += place_loss.item()
                total_action_batches += 1
                
                loss = action_loss + card_loss + place_loss
                
                # Collect card predictions for samples with action=1
                card_preds = card_logits_masked.argmax(dim=1)
                all_card_preds.extend(card_preds.cpu().tolist())
                all_card_labels.extend(labels_card_masked.cpu().tolist())
            else:
                loss = action_loss

            total_loss += loss.item()
            total_batches += 1

            # Collect action predictions (sigmoid > 0.5)
            action_preds = (torch.sigmoid(action_logits.squeeze(1)) > 0.5).long()
            all_action_preds.extend(action_preds.cpu().tolist())
            all_action_labels.extend(labels_action.cpu().tolist())

    avg_loss = total_loss / max(total_batches, 1)
    avg_action = total_action_loss / max(total_batches, 1)
    avg_card = total_card_loss / max(total_action_batches, 1) if total_action_batches > 0 else 0.0
    avg_place = total_place_loss / max(total_action_batches, 1) if total_action_batches > 0 else 0.0

    # Action metrics: binary classification (play vs no-play)
    action_precision = precision_score(all_action_labels, all_action_preds, pos_label=1, zero_division=0)
    action_recall = recall_score(all_action_labels, all_action_preds, pos_label=1, zero_division=0)
    action_f1 = f1_score(all_action_labels, all_action_preds, pos_label=1, zero_division=0)
    action_accuracy = sum(p == l for p, l in zip(all_action_preds, all_action_labels)) / max(len(all_action_preds), 1)
    
    # Card metrics: multi-class (only over samples where action=1)
    if all_card_labels:
        present_classes = sorted(set(all_card_labels))
        card_precision = precision_score(all_card_labels, all_card_preds, labels=present_classes, average='macro', zero_division=0)
        card_recall = recall_score(all_card_labels, all_card_preds, labels=present_classes, average='macro', zero_division=0)
        card_f1 = f1_score(all_card_labels, all_card_preds, labels=present_classes, average='macro', zero_division=0)
        card_accuracy = sum(p == l for p, l in zip(all_card_preds, all_card_labels)) / max(len(all_card_preds), 1)
    else:
        card_precision = card_recall = card_f1 = card_accuracy = 0.0

    # Print summary
    print(
        f"[Eval] {split_name} epoch {epoch + 1}: "
        f"loss={avg_loss:.4f}, action={avg_action:.4f}, card={avg_card:.4f}, place={avg_place:.4f}"
    )
    print(
        f"[Eval] {split_name} epoch {epoch + 1} (action head): "
        f"accuracy={action_accuracy:.4f}, precision={action_precision:.4f}, recall={action_recall:.4f}, f1={action_f1:.4f}"
    )
    print(
        f"[Eval] {split_name} epoch {epoch + 1} (card head): "
        f"accuracy={card_accuracy:.4f}, precision={card_precision:.4f}, recall={card_recall:.4f}, f1={card_f1:.4f}"
    )

    wandb.log(
        {
            f"{split_name}/loss": avg_loss,
            f"{split_name}/action_loss": avg_action,
            f"{split_name}/card_loss": avg_card,
            f"{split_name}/place_loss": avg_place,
            f"{split_name}/action_accuracy": action_accuracy,
            f"{split_name}/action_precision": action_precision,
            f"{split_name}/action_recall": action_recall,
            f"{split_name}/action_f1": action_f1,
            f"{split_name}/card_accuracy": card_accuracy,
            f"{split_name}/card_precision": card_precision,
            f"{split_name}/card_recall": card_recall,
            f"{split_name}/card_f1": card_f1,
        }
    )

    return avg_loss


def run_training(game_config, hyperparameter_config, runtime_config):
    with wandb.init(project="clash-royale-decision-model", config=hyperparameter_config):
        wandb_config = wandb.config
        config = dict(wandb_config)
        config.update(game_config)
        config.update(runtime_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Enable cuDNN auto-tuner for faster convolutions
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        train_loader, val_loader, test_loader, dataset = build_dataloaders(config, device)
        model = build_model(config, device)

        # Log basic model information
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {model.__class__.__name__}")
        print(f"Total parameters: {num_params:,}; trainable: {num_trainable:,}")

        # Compute pos_weight for balanced action loss
        action_pos_weight = compute_action_pos_weight(dataset, device)
        
        # Loss functions for two-head architecture
        action_loss_fn = nn.BCEWithLogitsLoss(pos_weight=action_pos_weight)  # Balanced binary loss
        card_loss_fn = nn.CrossEntropyLoss()     # Multi-class: which card
        place_loss_fn = nn.CrossEntropyLoss()    # Placement on grid

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.0),
        )

        scheduler = None
        if config.get("use_scheduler", False):
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["num_epochs"],
            )

        # Use automatic mixed precision for faster training and lower memory
        use_amp = config.get("use_amp", True) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) # type: ignore

        best_val_loss = float("inf")

        # Persistent rolling average dict across epochs
        rolling_losses = None

        for epoch in range(config["num_epochs"]):
            rolling_losses = train_one_epoch(
                model,
                train_loader,
                optimizer,
                action_loss_fn,
                card_loss_fn,
                place_loss_fn,
                device,
                config,
                epoch,
                rolling_losses,
                scaler=scaler,
                use_amp=use_amp,
            )

            val_loss = evaluate(
                model,
                val_loader,
                action_loss_fn,
                card_loss_fn,
                place_loss_fn,
                device,
                config,
                epoch,
                "val",
            )

            if scheduler is not None:
                scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(config.get("output_dir", "./checkpoints"), "best_model.pt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "config": dict(config)}, save_path)

        # final test evaluation
        test_loss = evaluate(
            model,
            test_loader,
            action_loss_fn,
            card_loss_fn,
            place_loss_fn,
            device,
            config,
            config["num_epochs"],
            "test",
        )


if __name__ == "__main__":
    # hyperparameters tracked by wandb
    game_config = {
        "num_cards": len(ALL_CARDS),
        "grid_h": 32,
        "grid_w": 18,
        "numeric_feat_dim": 1,
    }
    
    hyperparameter_config = {
        "batch_size": 16,
        "num_epochs": 5,
        "learning_rate": 6e-4,
        "weight_decay": 1e-2,
        "use_scheduler": True,
        "max_grad_norm": 1.0,
        "frames_per_sample": 1,
        "convlstm_hidden": 128,
        "backbone_proj_channels": 128,
        "backbone_pretrained": True,
    }

    # runtime-only parameters (not tracked by wandb)
    runtime_config = {
        "hf_repo_id": "your-username/your-parquet-repo",
        "hf_repo_type": "dataset",
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "log_every": 10,
        "rolling_average_window": 100,
        "output_dir": "./checkpoints",
    }

    run_training(game_config, hyperparameter_config, runtime_config)
