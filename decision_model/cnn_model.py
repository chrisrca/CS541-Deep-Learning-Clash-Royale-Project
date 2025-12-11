"""
Spatial Attention-based Card + Placement model for Clash Royale-style gameplay.
Single-frame input, no LSTM.

Requirements:
    torch, torchvision

Model summary:
    - Backbone: MobileNetV2 for efficient feature extraction
    - Spatial Attention: Transformer Encoder over flattened spatial features
    - Input:
        frames: (B, C, H, W) float tensor, normalized (single frame)
        numeric_feats: (B, numeric_feat_dim) float tensor
        playable_mask: (B, num_cards) boolean or 0/1 tensor
        hand_mask: (B, num_cards) boolean or 0/1 tensor
    - Outputs:
        action_logits: (B, 1)
        card_logits: (B, num_cards)
        placement_logits: (B, num_cards, grid_h * grid_w)
        placement_map: (B, num_cards, grid_h, grid_w)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from typing import Optional, Tuple

# -------------------------
# Backbone wrapper (MobileNetV2)
# -------------------------
class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 backbone that outputs spatial feature maps.
    Input expected: (B, C, H, W)
    Output:
        - spatial_map: (B, out_channels, Hf, Wf)
    """

    def __init__(self, weights: Optional[MobileNet_V2_Weights] = None, proj_out_channels: int = 128):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=weights)
        # keep only the features (convolutional layers)
        # MobileNetV2 features output 1280 channels
        self.features = mobilenet.features
        
        self.backbone_out_channels = 1280
        # optional projection to reduce channels
        if proj_out_channels is not None and proj_out_channels != self.backbone_out_channels:
            self.project = nn.Conv2d(self.backbone_out_channels, proj_out_channels, kernel_size=1)
            self.out_channels = proj_out_channels
        else:
            self.project = None
            self.out_channels = self.backbone_out_channels

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)
        x = self.features(x)  # (B, 1280, Hf, Wf)
        if self.project is not None:
            x = self.project(x)  # (B, proj_out_channels, Hf, Wf)
        return x

# -------------------------
# Full Model
# -------------------------
class CNNClashRoyaleModel(nn.Module):
    def __init__(
        self,
        *,
        num_cards: int,
        numeric_feat_dim: int,
        grid_h: int,
        grid_w: int,
        backbone_weights: Optional[MobileNet_V2_Weights] = None,
        backbone_proj_channels: int = 128,
    ):
        """
        Args:
            num_cards: number of card classes
            numeric_feat_dim: dimensionality of the numeric feature vector
            grid_h, grid_w: output placement grid resolution
            input_h, input_w: input image resolution (used to calc feature map size)
            backbone_weights: Pretrained weights for backbone
            backbone_proj_channels: Channel dimension after backbone projection
        """
        super().__init__()
        self.num_cards = num_cards
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.numeric_feat_dim = numeric_feat_dim

        # 1. Backbone
        self.backbone = MobileNetV2Backbone(weights=backbone_weights, proj_out_channels=backbone_proj_channels)
        self.feature_dim = self.backbone.out_channels

        # 2. Spatial Attention REMOVED
        # We now use the backbone features directly.

        # 3. Heads
        # Shared feature dimension for dense heads
        # We concatenate Global Pooled Features + Numeric + Hand
        shared_feat_dim = self.feature_dim + numeric_feat_dim + num_cards
        head_hidden = 256
        
        # Action head: binary classifier (play vs no-play)
        self.action_head_mlp = nn.Sequential(
            nn.LayerNorm(shared_feat_dim),
            nn.Linear(shared_feat_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        
        # Card head: multi-class classifier
        self.card_head_mlp = nn.Sequential(
            nn.LayerNorm(shared_feat_dim),
            nn.Linear(shared_feat_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, num_cards),
        )

        # Placement head:
        # Takes backbone features (B, C, Hf, Wf)
        # Concatenates numeric features (broadcasted)
        # Decodes to (B, num_cards, grid_h, grid_w)
        
        place_mid_ch = 128
        self.place_reduce_conv = nn.Sequential(
            nn.Conv2d(self.feature_dim, place_mid_ch, kernel_size=1),
            nn.BatchNorm2d(place_mid_ch),
            nn.GELU(),
        )

        fused_ch = place_mid_ch + numeric_feat_dim
        self.place_conv_block = nn.Sequential(
            nn.Conv2d(fused_ch, fused_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_ch),
            nn.GELU(),
            nn.Conv2d(fused_ch, fused_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_ch // 2),
            nn.GELU(),
        )
        self.place_logits = nn.Conv2d(fused_ch // 2, num_cards, kernel_size=1)

    def forward(self, frames: torch.Tensor, numeric_feats: torch.Tensor, playable_mask: torch.Tensor, hand_mask: torch.Tensor):
        """
        Args:
            frames: (B, C, H, W) - Single frame
            numeric_feats: (B, numeric_feat_dim)
            playable_mask: (B, num_cards)
            hand_mask: (B, num_cards)
        """
        B, C, H, W = frames.shape
        
        # 1. Extract Features
        # (B, C, Hf, Wf)
        features = self.backbone(frames)
        
        # 2. No Spatial Attention
        # Use features directly
        attended_features = features
        
        # 3. Global Pooling for Action/Card heads
        # (B, C)
        pooled_features = F.adaptive_avg_pool2d(attended_features, output_size=(1, 1)).view(B, -1)
        
        # 4. Prepare Shared Features
        # Concat: [Pooled Features, Numeric Features, Hand Mask]
        # Note: hand_mask is used as a feature here, similar to original model
        combined_features = torch.cat([pooled_features, numeric_feats, hand_mask], dim=1)
        
        # 5. Action & Card Heads
        action_logits = self.action_head_mlp(combined_features) # (B, 1)
        card_logits = self.card_head_mlp(combined_features)     # (B, num_cards)
        
        # Mask card logits
        if playable_mask is not None:
            # Set logits of unplayable cards to a very large negative number
            # playable_mask is 1.0 for playable, 0.0 for not
            # We want to keep original logits where mask is 1, and -inf where mask is 0
            # (1 - mask) * -1e9
            mask_penalty = (1.0 - playable_mask) * -1e9
            card_logits = card_logits + mask_penalty

        # 6. Placement Head
        # Use attended_features (B, C, Hf, Wf)
        x_place = self.place_reduce_conv(attended_features) # (B, mid_ch, Hf, Wf)
        
        # Broadcast numeric features to spatial grid
        # numeric_feats: (B, num_feat) -> (B, num_feat, 1, 1) -> (B, num_feat, Hf, Wf)
        Hf, Wf = x_place.shape[-2:]
        num_feat_spatial = numeric_feats.view(B, self.numeric_feat_dim, 1, 1).expand(-1, -1, Hf, Wf)
        
        x_place = torch.cat([x_place, num_feat_spatial], dim=1)
        x_place = self.place_conv_block(x_place)
        
        # (B, num_cards, Hf, Wf)
        place_logits_small = self.place_logits(x_place)
        
        # Resize to target grid size (grid_h, grid_w)
        # Using bilinear interpolation
        placement_map = F.interpolate(place_logits_small, size=(self.grid_h, self.grid_w), mode='bilinear', align_corners=False)
        
        # Flatten placement logits for loss computation: (B, num_cards, grid_h * grid_w)
        placement_logits_flat = placement_map.view(B, self.num_cards, -1)

        return {
            "action_logits": action_logits,
            "card_logits": card_logits,
            "placement_logits": placement_logits_flat,
            "placement_map": placement_map
        }
