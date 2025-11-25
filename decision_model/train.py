import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from huggingface_hub import list_repo_files, hf_hub_download
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

    parquet_paths = ["./initial_training_hand_elixir.parquet"]
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
    return train_loader, val_loader, test_loader


def build_model(config, device):
    # We add +1 to num_cards to account for the "No-Op" / "Wait" action
    num_cards = config["num_cards"] + 1
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


def train_one_epoch(model, train_loader, optimizer, card_loss_fn, place_loss_fn, device, config, epoch, rolling_batch_losses=None, rolling_card_losses=None, rolling_place_losses=None):
    model.train()
    total_loss = 0.0
    total_card_loss = 0.0
    total_place_loss = 0.0
    total_batches = 0

    # Store per-batch losses to compute rolling averages for logging.
    # Use persistent lists that carry over across epochs if provided
    if rolling_batch_losses is None:
        batch_losses = []
        batch_card_losses = []
        batch_place_losses = []
    else:
        batch_losses = rolling_batch_losses
        batch_card_losses = rolling_card_losses
        batch_place_losses = rolling_place_losses

    frames_per_sample = config["frames_per_sample"]

    num_epochs = config.get("num_epochs", None)
    if num_epochs is not None:
        print(f"Starting training epoch {epoch + 1}/{num_epochs}...")
    else:
        print(f"Starting training epoch {epoch + 1}...")

    for batch_idx, batch in enumerate(train_loader):
        frames = batch["frames"].to(device)
        numeric_features = batch["numeric_features"].to(device)
        labels_card = batch["label_card"].long().to(device)
        labels_placement = batch["label_placement"].long().to(device).view(-1)
        
        playable_mask = batch["playable_mask"].to(device)
        hand_mask = batch["hand_mask"].to(device)

        if frames.ndim == 4:
            frames = frames.unsqueeze(1)

        T = frames.shape[1]
        use_T = min(frames_per_sample, T)
        frames = frames[:, -use_T:, ...]

        optimizer.zero_grad(set_to_none=True)
        outputs = model(frames, numeric_features, playable_mask, hand_mask)
        card_logits = outputs["card_logits"]
        placement_logits = outputs["placement_logits"] # (B, num_cards, grid_cells)

        # Gather the placement logits for the ground-truth card
        # labels_card: (B,) containing the index of the card played
        B_dim = placement_logits.shape[0]
        # We want [B, grid_cells] from [B, num_cards, grid_cells]
        # using labels_card as the index for dim 1
        relevant_placement_logits = placement_logits[torch.arange(B_dim, device=device), labels_card, :]

        card_loss = card_loss_fn(card_logits, labels_card)
        place_loss = place_loss_fn(relevant_placement_logits, labels_placement)
        loss = card_loss + place_loss

        loss.backward()

        max_grad_norm = config.get("max_grad_norm", 0.0)
        if max_grad_norm and max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        total_card_loss += card_loss.item()
        total_place_loss += place_loss.item()
        total_batches += 1

        batch_losses.append(loss.item())
        batch_card_losses.append(card_loss.item())
        batch_place_losses.append(place_loss.item())

        if (batch_idx + 1) % config["log_every"] == 0:
            # Compute rolling average over the last N batches
            window = config["rolling_average_window"]
            start_idx = max(0, len(batch_losses) - window)
            window_losses = batch_losses[start_idx:]
            window_card_losses = batch_card_losses[start_idx:]
            window_place_losses = batch_place_losses[start_idx:]
            avg_window_loss = sum(window_losses) / max(len(window_losses), 1)
            avg_window_card = sum(window_card_losses) / max(len(window_card_losses), 1)
            avg_window_place = sum(window_place_losses) / max(len(window_place_losses), 1)

            print(
                f"[Train] epoch {epoch + 1}, batch {batch_idx + 1}/{len(train_loader)} "
                f"loss={avg_window_loss:.4f}, card={avg_window_card:.4f}, place={avg_window_place:.4f}"
            )
            wandb.log(
                {
                    "train/loss": avg_window_loss,
                    "train/card_loss": avg_window_card,
                    "train/place_loss": avg_window_place,
                }
            )

    avg_loss = total_loss / max(total_batches, 1)
    avg_card = total_card_loss / max(total_batches, 1)
    avg_place = total_place_loss / max(total_batches, 1)

    print(
        f"[Train] epoch {epoch + 1} completed: "
        f"avg loss={avg_loss:.4f}, avg card={avg_card:.4f}, avg place={avg_place:.4f}"
    )
    return avg_loss, avg_card, avg_place


def evaluate(model, data_loader, card_loss_fn, place_loss_fn, device, config, epoch, split_name):
    model.eval()
    total_loss = 0.0
    total_card_loss = 0.0
    total_place_loss = 0.0
    total_batches = 0

    # Accuracy metrics for card prediction
    total_samples = 0
    total_correct = 0

    frames_per_sample = config["frames_per_sample"]

    with torch.no_grad():
        for batch in data_loader:
            frames = batch["frames"].to(device)
            numeric_features = batch["numeric_features"].to(device)
            labels_card = batch["label_card"].long().to(device)
            labels_placement = batch["label_placement"].long().to(device).view(-1)
            
            playable_mask = batch["playable_mask"].to(device)
            hand_mask = batch["hand_mask"].to(device)

            if frames.ndim == 4:
                frames = frames.unsqueeze(1)

            T = frames.shape[1]
            use_T = min(frames_per_sample, T)
            frames = frames[:, -use_T:, ...]

            outputs = model(frames, numeric_features, playable_mask, hand_mask)
            card_logits = outputs["card_logits"]
            placement_logits = outputs["placement_logits"]

            # Select placement logits for the ground-truth card, to match
            # the training-time loss computation shape: [B, grid_cells]
            B_dim = placement_logits.shape[0]
            relevant_placement_logits = placement_logits[torch.arange(B_dim, device=device), labels_card, :]

            card_loss = card_loss_fn(card_logits, labels_card)
            place_loss = place_loss_fn(relevant_placement_logits, labels_placement)
            loss = card_loss + place_loss

            total_loss += loss.item()
            total_card_loss += card_loss.item()
            total_place_loss += place_loss.item()
            total_batches += 1

            # Accuracy for card prediction
            with torch.no_grad():
                preds = card_logits.argmax(dim=1)
                total_correct += (preds == labels_card).sum().item()

                total_samples += labels_card.shape[0]

    avg_loss = total_loss / max(total_batches, 1)
    avg_card = total_card_loss / max(total_batches, 1)
    avg_place = total_place_loss / max(total_batches, 1)

    # Compute accuracy
    accuracy = total_correct / max(total_samples, 1)

    # Print a concise summary line similar to training
    print(
        f"[Eval] {split_name} epoch {epoch + 1}: "
        f"loss={avg_loss:.4f}, card={avg_card:.4f}, place={avg_place:.4f}, "
        f"accuracy={accuracy:.4f}"
    )

    wandb.log(
        {
            f"{split_name}/loss": avg_loss,
            f"{split_name}/card_loss": avg_card,
            f"{split_name}/place_loss": avg_place,
            f"{split_name}/accuracy": accuracy,
        }
    )

    return avg_loss, avg_card, avg_place


def run_training(game_config, hyperparameter_config, runtime_config):
    with wandb.init(project="clash-royale-decision-model", config=hyperparameter_config):
        wandb_config = wandb.config
        config = dict(wandb_config)
        config.update(game_config)
        config.update(runtime_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        train_loader, val_loader, test_loader = build_dataloaders(config, device)
        model = build_model(config, device)

        # Log basic model information
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {model.__class__.__name__}")
        print(f"Total parameters: {num_params:,}; trainable: {num_trainable:,}")

        card_loss_fn = nn.CrossEntropyLoss()
        place_loss_fn = nn.CrossEntropyLoss(ignore_index=-1) # No placement loss when no card was played

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

        best_val_loss = float("inf")

        # Persistent rolling average lists across epochs
        rolling_batch_losses = []
        rolling_card_losses = []
        rolling_place_losses = []

        for epoch in range(config["num_epochs"]):
            train_loss, train_card, train_place = train_one_epoch(
                model,
                train_loader,
                optimizer,
                card_loss_fn,
                place_loss_fn,
                device,
                config,
                epoch,
                rolling_batch_losses,
                rolling_card_losses,
                rolling_place_losses,
            )

            val_loss, val_card, val_place = evaluate(
                model,
                val_loader,
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
        test_loss, test_card, test_place = evaluate(
            model,
            test_loader,
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
        "numeric_feat_dim": 7,
    }
    
    hyperparameter_config = {
        "batch_size": 16,
        "num_epochs": 10,
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
        "log_every": 1,
        "rolling_average_window": 100,
        "output_dir": "./checkpoints",
    }

    run_training(game_config, hyperparameter_config, runtime_config)
