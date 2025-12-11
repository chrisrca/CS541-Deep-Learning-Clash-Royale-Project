import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os
import sys
from PIL import Image as PILImage

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_model.model import ConvLSTMClashRoyaleModel
from decision_model.dataset import ClashRoyaleDataset, ALL_CARDS, Y_OFFSET_TOP, Y_OFFSET_BOTTOM, X_OFFSET_LEFT, X_OFFSET_RIGHT

# Configuration (must match training config)
config = {
    "num_cards": len(ALL_CARDS),
    "grid_h": 32,
    "grid_w": 18,
    "numeric_feat_dim": 1,
    "convlstm_hidden": 64,
    "backbone_proj_channels": 64,
    "transformer_layers": 2,
    "transformer_heads": 4,
    "transformer_dropout": 0.1,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
}

# Create ID to card name mapping
ID_TO_CARD = {i: name for i, name in enumerate(sorted(ALL_CARDS))}
CARD_TO_ID = {name: i for i, name in enumerate(sorted(ALL_CARDS))}

def load_model(checkpoint_path, config, device):
    """Load the trained model from checkpoint."""
    num_cards = config["num_cards"]
    
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
    """Load dataset and get a random sample from validation set."""
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
    
    # Get a random sample from validation set
    idx = np.random.randint(0, len(val_dataset))
    sample = val_dataset[idx]
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
    
    with torch.no_grad():
        outputs = model(frames, numeric_features, playable_mask, hand_mask)
    
    return outputs

def visualize_full_output(sample, outputs, config):
    """Display all model outputs: Action Prob, Card Probs, Gameplay, and Placement Heatmap."""
    
    # --- 1. Prepare Data ---
    
    # Action Probability
    action_logits = outputs["action_logits"].cpu().numpy().flatten()
    action_prob = 1 / (1 + np.exp(-action_logits[0]))
    
    # Card Probabilities
    card_logits = outputs["card_logits"].cpu().numpy().flatten()
    card_probs = np.exp(card_logits) / np.sum(np.exp(card_logits))
    
    # Ground Truth Info
    label_action = sample["label_action"].item()
    label_card = sample["label_card"].item()
    ground_truth_card = ID_TO_CARD.get(label_card, "unknown") if label_action == 1 else "None"
    
    # Hand Info
    hand_mask = sample["hand_mask"].cpu().numpy()
    cards_in_hand_indices = np.where(hand_mask == 1)[0]
    
    # Get top prediction to decide which heatmap to show
    predicted_card_idx = np.argmax(card_probs)
    predicted_card_name = ID_TO_CARD.get(predicted_card_idx, "unknown")

    # If we have ground truth, show that heatmap, otherwise show predicted
    # Actually, let's show the predicted card's heatmap if action prob > 0.5, else maybe ground truth?
    # Let's prioritize showing the heatmap for the *predicted* card if valid, otherwise the ground truth.
    # If ground truth exists (played a card), let's show that.
    
    target_heatmap_card_idx = label_card if label_action == 1 else predicted_card_idx
    target_heatmap_card_name = ID_TO_CARD.get(target_heatmap_card_idx, "unknown")

    hand_card_probs = card_probs[cards_in_hand_indices]
    hand_card_names = [ID_TO_CARD[i] for i in cards_in_hand_indices]
    
    if len(cards_in_hand_indices) == 0:
         # Fallback
        top_indices = np.argsort(card_probs)[::-1][:4]
        hand_card_probs = card_probs[top_indices]
        hand_card_names = [ID_TO_CARD[i] for i in top_indices]
        cards_in_hand_indices = top_indices

    # Sort by probability for display
    sorted_order = np.argsort(hand_card_probs)[::-1]
    sorted_probs = hand_card_probs[sorted_order]
    sorted_names = [hand_card_names[i] for i in sorted_order]
    sorted_indices = cards_in_hand_indices[sorted_order]
    
    # Gameplay Image
    frames = sample["frames"]
    frame = frames[0] if frames.ndim == 4 else frames
    gameplay_img = frame.permute(1, 2, 0).numpy()
    img_height, img_width = gameplay_img.shape[:2]
    
    # Heatmap
    placement_map = outputs["placement_map"][0] # (num_cards, grid_h, grid_w)
    heatmap = placement_map[target_heatmap_card_idx].cpu().numpy()
    heatmap_softmax = torch.softmax(torch.tensor(heatmap.flatten()), dim=0).numpy()
    heatmap_softmax = heatmap_softmax.reshape(config["grid_h"], config["grid_w"])
    
    # Create padded heatmap overlay
    padded_heatmap = np.zeros((img_height, img_width))
    grid_pixel_height = img_height - Y_OFFSET_TOP - Y_OFFSET_BOTTOM
    grid_pixel_width = img_width - X_OFFSET_LEFT - X_OFFSET_RIGHT
    
    heatmap_resized = np.array(PILImage.fromarray(heatmap_softmax.astype(np.float32)).resize(
        (grid_pixel_width, grid_pixel_height), PILImage.Resampling.NEAREST
    ))
    padded_heatmap[Y_OFFSET_TOP:Y_OFFSET_TOP + grid_pixel_height, 
                   X_OFFSET_LEFT:X_OFFSET_LEFT + grid_pixel_width] = heatmap_resized

    # --- 2. Plotting ---
    # Layout: 
    # [ Gameplay (Left) ] [ Heatmap (Center) ] [ Action Prob (Top Right) ]
    #                                          [ Card Probs (Bottom Right) ]
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
    
    # Left: Gameplay (spans both rows)
    ax_game = fig.add_subplot(gs[:, 0])
    
    # Center: Heatmap (spans both rows)
    ax_heat = fig.add_subplot(gs[:, 1])
    
    # Top Right: Action Probability
    ax_action = fig.add_subplot(gs[0, 2])
    
    # Bottom Right: Card Probabilities
    ax_cards = fig.add_subplot(gs[1, 2])
    
    # 1. Gameplay Image
    ax_game.imshow(gameplay_img)
    ax_game.set_title("Input Frame")
    ax_game.axis("off")
    
    # 2. Placement Heatmap
    # Display on black background (padded_heatmap)
    # We want it to match ax_game size exactly.
    # The colorbar often shrinks the axis. Let's put the colorbar horizontally below or use a specific method.
    
    im = ax_heat.imshow(padded_heatmap, cmap="hot", vmin=0)
    ax_heat.set_title(f"Placement Heatmap for: {target_heatmap_card_name}")
    ax_heat.axis("off")
    
    # Add colorbar horizontally at the bottom of the heatmap axes
    # Using inset_axes to place colorbar inside/near the plot without resizing it
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax_heat,
                       width="50%",  # width = 50% of parent_bbox width
                       height="3%",  # height : 5%
                       loc='lower center',
                       bbox_to_anchor=(0, -0.1, 1, 1),
                       bbox_transform=ax_heat.transAxes,
                       borderpad=0,
                       )
    plt.colorbar(im, cax=axins, orientation="horizontal")
    
    ax_heat.text(
        img_width / 2, img_height / 2, 
        f"Showing Map For:\n{target_heatmap_card_name}",
        ha="center", va="center", fontsize=12, color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7)
    )

    # 3. Action Probability
    ax_action.bar(["Action Probability"], [action_prob], color='skyblue', width=0.3)
    ax_action.set_ylim(0, 1.1)
    ax_action.set_ylabel("Probability")
    ax_action.set_title(f"Action: {'Play' if action_prob > 0.5 else 'No-Op'}\n(GT: {'Play' if label_action == 1 else 'No-Op'})")
    ax_action.axhline(y=0.5, color='r', linestyle='--', label="Threshold (0.5)")
    ax_action.text(0, action_prob + 0.02, f"{action_prob:.4f}", ha='center', fontweight='bold')
    ax_action.legend()
    
    # 4. Card Probabilities
    colors = ['green' if idx == label_card else 'steelblue' for idx in sorted_indices]
    ax_cards.bar(sorted_names, sorted_probs, color=colors)
    ax_cards.set_ylabel("Probability")
    ax_cards.set_title(f"Card Selection (In Hand)\n(GT: {ground_truth_card})")
    ax_cards.tick_params(axis='x', rotation=45, labelsize=10)
    ax_cards.set_ylim(0, 1.0)
    for i, prob in enumerate(sorted_probs):
        ax_cards.text(i, prob + 0.01, f"{prob:.2f}", ha='center', va='bottom', fontsize=9)

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
    
    print("Loading random validation sample...")
    try:
        sample = get_validation_sample(config)
    except Exception as e:
        print(f"Error loading sample: {e}")
        return

    print("Sample loaded.")
    
    # Run inference
    print("Running inference...")
    outputs = run_inference(model, sample, device)
    print("Inference complete.")
    
    # Visualize
    print("Displaying visualization...")
    visualize_full_output(sample, outputs, config)

if __name__ == "__main__":
    main()
