import glob
import os
from bisect import bisect_right

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from huggingface_hub import list_repo_files, hf_hub_download
import wandb
from model import ConvLSTMCardPlacementModel

grid_w = 18
grid_h = 32


class ParquetClashDataset(Dataset):
    def __init__(self, files):
        super().__init__()
        self.files = list(files)
        if not self.files:
            raise ValueError("No parquet files provided to ParquetClashDataset")

        self.row_counts = []
        self.col_counts = []
        total = 0
        for path in self.files:
            pf = pq.ParquetFile(path)
            n_rows = pf.metadata.num_rows
            self.row_counts.append(n_rows)
            total += n_rows
            self.col_counts.append(total)

    def __len__(self) -> int:
        return self.col_counts[-1]

    def _locate_row(self, idx: int):
        # returns file index and row index within that file
        file_idx = bisect_right(self.col_counts, idx)
        prev_cum = 0 if file_idx == 0 else self.col_counts[file_idx - 1]
        row_idx = idx - prev_cum
        return file_idx, row_idx

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        file_idx, row_idx = self._locate_row(idx)
        path = self.files[file_idx]

        table = pq.read_table(path)
        # slice out the single row we need
        row_table = table.slice(row_idx, 1)
        data = row_table.to_pydict()

        frames_arr = np.array(data["frames"][0])
        frames = torch.from_numpy(frames_arr.astype(np.float32))

        if frames.ndim == 4 and frames.shape[-1] in (1, 3):
            frames = frames.permute(0, 3, 1, 2)

        frames = frames / 255.0

        card = int(data["card"][0])
        tile_x = int(data["tile_x"][0])
        tile_y = int(data["tile_y"][0])

        elixir = float(data["elixir"][0])

        blue_left_princess_tower_health = int(data["blue_left_princess_tower_health"][0])
        blue_right_princess_tower_health = int(data["blue_right_princess_tower_health"][0])
        blue_king_tower_health = int(data["blue_king_tower_health"][0])
        red_left_princess_tower_health = int(data["red_left_princess_tower_health"][0])
        red_right_princess_tower_health = int(data["red_right_princess_tower_health"][0])
        red_king_tower_health = int(data["red_king_tower_health"][0])

        extra_features = torch.tensor([elixir, blue_left_princess_tower_health, blue_right_princess_tower_health, blue_king_tower_health, red_left_princess_tower_health, red_right_princess_tower_health, red_king_tower_health], dtype=torch.float32)

        label_card = card
        tile_index = tile_y * grid_w + tile_x
        label_placement = torch.tensor(tile_index, dtype=torch.float32)

        return {
            "frames": frames,
            "extra_features": extra_features,
            "label_card": label_card,
            "label_placement": label_placement,
        }


def get_hf_parquet_local_paths(repo_id: str, repo_type: str = "dataset"):
    """List all .parquet files in a Hugging Face repo and download them locally.

    Returns a list of local cached file paths suitable for ParquetClashDataset.
    """
    files = list_repo_files(repo_id, repo_type=repo_type)
    parquet_files = [f for f in files if f.endswith(".parquet")]
    if not parquet_files:
        raise ValueError(f"No .parquet files found in HF repo: {repo_id}")

    local_paths = []
    for fp in parquet_files:
        local_path = hf_hub_download(repo_id=repo_id, filename=fp, repo_type=repo_type)
        local_paths.append(local_path)
    return local_paths


def build_dataloaders(config, device):
    hf_repo_id = config["hf_repo_id"]
    hf_repo_type = config.get("hf_repo_type", "dataset")

    parquet_paths = get_hf_parquet_local_paths(hf_repo_id, repo_type=hf_repo_type)
    dataset = ParquetClashDataset(parquet_paths)

    val_ratio = config.get("val_ratio", 0.1)
    test_ratio = config.get("test_ratio", 0.1)
    assert 0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0 and val_ratio + test_ratio < 1.0

    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 4),
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 4),
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 4),
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_loader, test_loader


def build_model(config, device):
    num_cards = config["num_cards"]
    extra_feat_dim = config["extra_feat_dim"]
    convlstm_hidden = config["convlstm_hidden"]
    backbone_proj_channels = config["backbone_proj_channels"]
    use_pretrained = config.get("backbone_pretrained", False)
    pretrained_weights = None
    if use_pretrained:
        from torchvision.models import MobileNet_V2_Weights

        pretrained_weights = MobileNet_V2_Weights.DEFAULT

    model = ConvLSTMCardPlacementModel(
        num_cards=num_cards,
        extra_feat_dim=extra_feat_dim,
        grid_h=grid_h,
        grid_w=grid_w,
        convlstm_hidden=convlstm_hidden,
        pretrained_backbone_weights=pretrained_weights,
        backbone_proj_channels=backbone_proj_channels,
    )
    return model.to(device)


def train_one_epoch(model, train_loader, optimizer, card_loss_fn, place_loss_fn, device, config, epoch):
    model.train()
    total_loss = 0.0
    total_card_loss = 0.0
    total_place_loss = 0.0
    total_batches = 0

    frames_per_sample = config["frames_per_sample"]

    for batch_idx, batch in enumerate(train_loader):
        frames = batch["frames"].to(device)
        extra_features = batch["extra_features"].to(device)
        labels_card = batch["label_card"].long().to(device)
        labels_placement = batch["label_placement"].long().to(device).view(-1)

        if frames.ndim == 4:
            frames = frames.unsqueeze(1)

        T = frames.shape[1]
        use_T = min(frames_per_sample, T)
        frames = frames[:, -use_T:, ...]

        optimizer.zero_grad(set_to_none=True)
        outputs = model(frames, extra_features)
        card_logits = outputs["card_logits"]
        placement_logits = outputs["placement_logits"]

        card_loss = card_loss_fn(card_logits, labels_card)
        place_loss = place_loss_fn(placement_logits, labels_placement)
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

        if (batch_idx + 1) % config.get("log_every", 100) == 0:
            wandb.log(
                {
                    "train/loss": total_loss / total_batches,
                    "train/card_loss": total_card_loss / total_batches,
                    "train/place_loss": total_place_loss / total_batches,
                    "train/epoch": epoch,
                    "train/step": epoch * len(train_loader) + batch_idx,
                }
            )

    avg_loss = total_loss / max(total_batches, 1)
    avg_card = total_card_loss / max(total_batches, 1)
    avg_place = total_place_loss / max(total_batches, 1)
    return avg_loss, avg_card, avg_place


def evaluate(model, data_loader, card_loss_fn, place_loss_fn, device, config, epoch, split_name):
    model.eval()
    total_loss = 0.0
    total_card_loss = 0.0
    total_place_loss = 0.0
    total_batches = 0

    frames_per_sample = config["frames_per_sample"]

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            frames = batch["frames"].to(device)
            extra_features = batch["extra_features"].to(device)
            labels_card = batch["label_card"].long().to(device)
            labels_placement = batch["label_placement"].long().to(device).view(-1)

            if frames.ndim == 4:
                frames = frames.unsqueeze(1)

            T = frames.shape[1]
            use_T = min(frames_per_sample, T)
            frames = frames[:, -use_T:, ...]

            outputs = model(frames, extra_features)
            card_logits = outputs["card_logits"]
            placement_logits = outputs["placement_logits"]

            card_loss = card_loss_fn(card_logits, labels_card)
            place_loss = place_loss_fn(placement_logits, labels_placement)
            loss = card_loss + place_loss

            total_loss += loss.item()
            total_card_loss += card_loss.item()
            total_place_loss += place_loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    avg_card = total_card_loss / max(total_batches, 1)
    avg_place = total_place_loss / max(total_batches, 1)

    wandb.log(
        {
            f"{split_name}/loss": avg_loss,
            f"{split_name}/card_loss": avg_card,
            f"{split_name}/place_loss": avg_place,
            f"{split_name}/epoch": epoch,
        }
    )

    return avg_loss, avg_card, avg_place


def run_training(train_config, runtime_config):
    with wandb.init(config=train_config):
        wandb_config = wandb.config
        config = dict(wandb_config)
        config.update(runtime_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loader, test_loader = build_dataloaders(config, device)
        model = build_model(config, device)

        card_loss_fn = nn.CrossEntropyLoss()
        place_loss_fn = nn.CrossEntropyLoss()

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
    train_config = {
        "batch_size": 16,
        "num_epochs": 5,
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "use_scheduler": True,
        "max_grad_norm": 1.0,
        "frames_per_sample": 4,
        "num_cards": 8,
        "extra_feat_dim": 7,
        "convlstm_hidden": 128,
        "backbone_proj_channels": 128,
        "backbone_pretrained": False,
    }

    # runtime-only parameters (not tracked by wandb)
    runtime_config = {
        "hf_repo_id": "your-username/your-parquet-repo",
        "hf_repo_type": "dataset",
        "num_workers": 4,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "log_every": 100,
        "output_dir": "./checkpoints",
    }

    run_training(train_config, runtime_config)
