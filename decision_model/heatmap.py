import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from model import ConvLSTMClashRoyaleModel
from dataset import ClashRoyaleDataset, ALL_CARDS, Y_OFFSET_TOP, Y_OFFSET_BOTTOM, X_OFFSET_LEFT, X_OFFSET_RIGHT

# Configuration (must match training config)
config = {
    "num_cards": len(ALL_CARDS),
    "grid_h": 32,
    "grid_w": 18,
    "numeric_feat_dim": 7,
    "convlstm_hidden": 128,
    "backbone_proj_channels": 128,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
}

# Create ID to card name mapping
ID_TO_CARD = {i: name for i, name in enumerate(sorted(ALL_CARDS))}
ID_TO_CARD[len(ALL_CARDS)] = "none"  # No-Op action

def load_model(checkpoint_path, config, device):
    """Load the trained model from checkpoint."""
    num_cards = config["num_cards"] + 1  # +1 for No-Op
    
    model = ConvLSTMClashRoyaleModel(
        num_cards=num_cards,
        numeric_feat_dim=config["numeric_feat_dim"],
        grid_h=config["grid_h"],
        grid_w=config["grid_w"],
        convlstm_hidden=config["convlstm_hidden"],
        pretrained_backbone_weights=None,  # Not needed for inference
        backbone_proj_channels=config["backbone_proj_channels"],
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model

def get_validation_sample(config):
    """Load dataset and get the first sample from validation set."""
    parquet_paths = ["./new_arena_placement.parquet"]
    dataset = ClashRoyaleDataset(parquet_paths, config["grid_w"], config["grid_h"], config["num_cards"])
    
    # Split dataset the same way as training
    val_ratio = config.get("val_ratio", 0.1)
    test_ratio = config.get("test_ratio", 0.1)
    
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    
    # Use the same random split (with same seed if needed for reproducibility)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test])
    
    # Get first sample from validation set
    sample = val_dataset[0]
    return sample

def run_inference(model, sample, device):
    """Run model inference on a single sample."""
    # Add batch dimension
    frames = sample["frames"].unsqueeze(0).to(device)
    numeric_features = sample["numeric_features"].unsqueeze(0).to(device)
    playable_mask = sample["playable_mask"].unsqueeze(0).to(device)
    hand_mask = sample["hand_mask"].unsqueeze(0).to(device)
    
    # Ensure frames has time dimension
    if frames.ndim == 4:
        frames = frames.unsqueeze(1)
    
    print(f"  Frames shape: {frames.shape}")
    print(f"  Device: {device}")
    
    with torch.no_grad():
        print("  Starting forward pass...")
        outputs = model(frames, numeric_features, playable_mask, hand_mask)
        print("  Forward pass complete.")
    
    return outputs

def visualize_heatmap(sample, outputs, config):
    """Display gameplay image and placement heatmap side by side."""
    # Get the gameplay image (first frame)
    frames = sample["frames"]
    if frames.ndim == 4:
        frame = frames[0]  # (C, H, W)
    else:
        frame = frames  # (C, H, W)
    
    # Convert to numpy for display (C, H, W) -> (H, W, C)
    gameplay_img = frame.permute(1, 2, 0).numpy()
    
    # Get ground truth card
    label_card = sample["label_card"].item()
    ground_truth_card = ID_TO_CARD.get(label_card, "unknown")
    
    # Get placement heatmap for the ground truth card
    placement_map = outputs["placement_map"][0]  # (num_cards, grid_h, grid_w)
    
    # Use the ground truth card's placement map
    heatmap = placement_map[label_card].cpu().numpy()  # (grid_h, grid_w)
    
    # Apply softmax to get probabilities
    heatmap_softmax = torch.softmax(torch.tensor(heatmap.flatten()), dim=0).numpy()
    heatmap_softmax = heatmap_softmax.reshape(config["grid_h"], config["grid_w"])
    
    # Get image dimensions
    img_height, img_width = gameplay_img.shape[:2]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Left: Gameplay image
    axes[0].imshow(gameplay_img)
    axes[0].set_title("Gameplay")
    axes[0].axis("off")
    
    # Right: Heatmap with proper padding
    # Create a padded heatmap that aligns with the gameplay area
    padded_heatmap = np.zeros((img_height, img_width))
    
    # Calculate the grid area in pixel coordinates
    grid_pixel_height = img_height - Y_OFFSET_TOP - Y_OFFSET_BOTTOM
    grid_pixel_width = img_width - X_OFFSET_LEFT - X_OFFSET_RIGHT
    
    # Resize heatmap to match the grid area in pixels
    from PIL import Image as PILImage
    heatmap_resized = np.array(PILImage.fromarray(heatmap_softmax.astype(np.float32)).resize(
        (grid_pixel_width, grid_pixel_height), PILImage.Resampling.NEAREST
    ))
    
    # Place the resized heatmap in the correct position
    padded_heatmap[Y_OFFSET_TOP:Y_OFFSET_TOP + grid_pixel_height, 
                   X_OFFSET_LEFT:X_OFFSET_LEFT + grid_pixel_width] = heatmap_resized
    
    # Display heatmap
    im = axes[1].imshow(padded_heatmap, cmap="hot", vmin=0)
    axes[1].set_title("Placement Heatmap")
    axes[1].axis("off")
    
    # Add ground truth card text in the middle of the heatmap
    axes[1].text(
        img_width / 2, 
        img_height / 2, 
        f"Ground Truth: {ground_truth_card}",
        ha="center", 
        va="center",
        fontsize=14,
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7)
    )
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = "./checkpoints/best_model.pt"
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, config, device)
    print("Model loaded successfully.")
    
    # Get validation sample
    print("Loading validation sample...")
    sample = get_validation_sample(config)
    print("Sample loaded.")
    
    # Run inference
    print("Running inference...")
    outputs = run_inference(model, sample, device)
    print("Inference complete.")
    
    # Visualize
    print("Displaying visualization...")
    visualize_heatmap(sample, outputs, config)

if __name__ == "__main__":
    main()
