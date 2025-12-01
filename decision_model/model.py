"""
ConvLSTM-based Card + Placement model for Clash Royale-style gameplay.

Requirements:
    torch, torchvision

Model summary:
    - Backbone: MobileNetV2 for efficient feature extraction
    - Input:
        frames: (B, T, C, H, W) float tensor, normalized to backbone's expectation
        numeric_feats: (B, numeric_feat_dim) float tensor (per-sample, only current frame info)
        playable_mask: (B, num_cards) boolean or 0/1 tensor where True/1 means legal to play
        hand_mask: (B, num_cards) boolean or 0/1 tensor where True/1 means in hand
    - Outputs:
        action_logits: (B, 1) raw logits for binary action decision (sigmoid -> probability of playing)
        card_logits: (B, num_cards) raw logits for which card to play (masked by playable_mask)
        placement_logits: (B, num_cards, grid_h * grid_w) raw logits over grid cells per card
        placement_map: (B, num_cards, grid_h, grid_w) raw logits reshaped
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from typing import Optional, Tuple


# -------------------------
# ConvLSTM cell / layer
# -------------------------
class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell.
    Input: x_t (B, C_in, H, W), h_prev (B, C_h, H, W), c_prev (B, C_h, H, W)
    Output: h_next, c_next (same spatial size as inputs)
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # gates: i, f, g, o
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        # x: (B, C_in, H, W)
        # hx: (h_prev, c_prev) where each is (B, C_h, H, W), or None
        if hx is None:
            batch, _, H, W = x.shape
            h_prev = torch.zeros(batch, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(batch, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = hx

        combined = torch.cat([x, h_prev], dim=1)  # (B, C_in + C_h, H, W)
        conv_out = self.conv(combined)  # (B, 4*C_h, H, W)
        i, f, g, o = torch.chunk(conv_out, chunks=4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    A simple ConvLSTM layer that runs over time and returns the sequence of h states (and final (h,c)).
    - input: (B, T, C_in, H, W)
    - output: h_seq (B, T, C_h, H, W), (h_last, c_last)
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels=input_channels, hidden_channels=hidden_channels,
                                 kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # x: (B, T, C_in, H, W)
        B, T, C_in, H, W = x.shape
        h_seq = []
        state = hx

        for t in range(T):
            xt = x[:, t]  # (B, C_in, H, W)
            h, c = self.cell(xt, state)
            state = (h, c)
            h_seq.append(h)

        h_seq = torch.stack(h_seq, dim=1)  # (B, T, C_h, H, W)
        return h_seq, (h, c) # type: ignore


# -------------------------
# Backbone wrapper (MobileNetV2)
# -------------------------
class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 backbone that outputs per-frame spatial feature maps.
    We'll drop the classifier and optionally project channels down.
    Input expected: (B*T, C, H, W)
    Output:
        - spatial_map: (B*T, out_channels, Hf, Wf)
    """

    def __init__(self, weights: Optional[MobileNet_V2_Weights] = None, proj_out_channels: int = 128):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=weights)
        # keep only the features (convolutional layers)
        # MobileNetV2 features output 1280 channels
        self.features = mobilenet.features
        
        self.backbone_out_channels = 1280
        # optional projection to reduce channels for ConvLSTM
        if proj_out_channels is not None and proj_out_channels != self.backbone_out_channels:
            self.project = nn.Conv2d(self.backbone_out_channels, proj_out_channels, kernel_size=1)
            self.out_channels = proj_out_channels
        else:
            self.project = None
            self.out_channels = self.backbone_out_channels

    def forward(self, x: torch.Tensor):
        # x: (B*T, C, H, W)
        x = self.features(x)  # (B*T, 1280, Hf, Wf)
        if self.project is not None:
            x = self.project(x)  # (B*T, proj_out_channels, Hf, Wf)
        return x


# -------------------------
# Full Model
# -------------------------
class ConvLSTMClashRoyaleModel(nn.Module):
    def __init__(
        self,
        *,
        num_cards: int,
        numeric_feat_dim: int,
        grid_h: int,
        grid_w: int,
        convlstm_hidden: int = 128,
        pretrained_backbone_weights: Optional[MobileNet_V2_Weights] = None,
        backbone_proj_channels: int = 128,
        convlstm_kernel: int = 3,
    ):
        """
        Args:
            num_cards: number of card classes
            numeric_feat_dim: dimensionality of the per-frame numeric feature vector (elixir, tower HPs, etc.)
            grid_h, grid_w: output placement grid resolution
            convlstm_hidden: hidden channels for ConvLSTM
            pretrained_backbone_weights: MobileNet_V2_Weights to use (e.g., MobileNet_V2_Weights.DEFAULT) or None for no pretrained weights
            backbone_proj_channels: project backbone channels down to this before ConvLSTM
        """
        super().__init__()
        self.num_cards = num_cards
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.grid_cells = grid_h * grid_w
        self.numeric_feat_dim = numeric_feat_dim

        # backbone
        self.backbone = MobileNetV2Backbone(weights=pretrained_backbone_weights, proj_out_channels=backbone_proj_channels)
        backbone_out_ch = self.backbone.out_channels

        # ConvLSTM
        self.convlstm = ConvLSTM(input_channels=backbone_out_ch, hidden_channels=convlstm_hidden,
                                 kernel_size=convlstm_kernel, padding=convlstm_kernel // 2)

        # Shared feature dimension for both heads
        shared_feat_dim = convlstm_hidden + numeric_feat_dim + num_cards  # +num_cards for hand_mask
        head_hidden = max(128, convlstm_hidden)
        
        # Action head: binary classifier (play vs no-play)
        # global pooling of final hidden map -> vector (B, convlstm_hidden)
        # then concat numeric_feats (B, numeric_feat_dim) and hand_mask (B, num_cards) -> MLP -> 1 logit
        self.action_head_mlp = nn.Sequential(
            nn.LayerNorm(shared_feat_dim),
            nn.Linear(shared_feat_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),  # Single logit, use BCE loss
        )
        
        # Card head: multi-class classifier for which card to play
        # Same input as action head -> MLP -> num_cards logits
        self.card_head_mlp = nn.Sequential(
            nn.LayerNorm(shared_feat_dim),
            nn.Linear(shared_feat_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, num_cards),
        )

        # placement head:
        # take final hidden map (B, convlstm_hidden, Hf, Wf), concat broadcasted numeric_feats,
        # a small conv decoder and finally produce grid logits
        # We'll first reduce convlstm_hidden -> mid channels, concat numeric_feat broadcast as channels,
        # then use conv layers -> produce 1-channel logits -> resize to (grid_h, grid_w)
        place_mid_ch = 128
        self.place_reduce_conv = nn.Sequential(
            nn.Conv2d(convlstm_hidden, place_mid_ch, kernel_size=1),
            nn.BatchNorm2d(place_mid_ch),
            nn.GELU(),
        )

        # after broadcasting numeric_feats, we'll have (place_mid_ch + numeric_feat_dim) channels
        fused_ch = place_mid_ch + numeric_feat_dim
        self.place_conv_block = nn.Sequential(
            nn.Conv2d(fused_ch, fused_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_ch),
            nn.GELU(),
            nn.Conv2d(fused_ch, fused_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_ch // 2),
            nn.GELU(),
        )
        # final logits per spatial location
        # We output 'num_cards' channels: one placement map for each card type.
        self.place_logits = nn.Conv2d(fused_ch // 2, num_cards, kernel_size=1)

    def forward(self, frames: torch.Tensor, numeric_feats: torch.Tensor, playable_mask: torch.Tensor, hand_mask: torch.Tensor):
        """
        Args:
            frames: (B, T, C, H, W)
            numeric_feats: (B, numeric_feat_dim)
            playable_mask: (B, num_cards) boolean or 0/1 tensor where True=legal to play
            hand_mask: (B, num_cards) boolean or 0/1 tensor where True=in hand
        Returns:
            dict with:
             - action_logits: (B, 1) binary logit for play vs no-play
             - card_logits: (B, num_cards) masked by playable_mask
             - placement_logits: (B, num_cards, grid_h * grid_w)
             - placement_map: (B, num_cards, grid_h, grid_w)
             - convlstm_hidden_seq: (B, T, convlstm_hidden, Hf, Wf)
        """
        B, T, C, H, W = frames.shape
        device = frames.device

        # 1) run backbone per-frame
        # Use reshape to ensure contiguous memory layout if needed
        frames_flat = frames.reshape(B * T, C, H, W)
        spatial_flat = self.backbone(frames_flat)  # (B*T, backbone_out_ch, Hf, Wf)
        _, backbone_ch, Hf, Wf = spatial_flat.shape

        # reshape to (B, T, C, Hf, Wf)
        spatial_seq = spatial_flat.reshape(B, T, backbone_ch, Hf, Wf)

        # 2) run ConvLSTM over spatial_seq
        h_seq, (h_last, c_last) = self.convlstm(spatial_seq)  # h_seq: (B, T, convlstm_hidden, Hf, Wf)
        # h_last: (B, convlstm_hidden, Hf, Wf)

        # 3) Shared feature computation
        # global-pool h_last -> (B, convlstm_hidden)
        pooled = F.adaptive_avg_pool2d(h_last, output_size=(1, 1)).view(B, -1)  # (B, convlstm_hidden)
        # concat numeric feats and hand_mask
        shared_input = torch.cat([pooled, numeric_feats, hand_mask], dim=1)  # (B, convlstm_hidden + numeric_feat_dim + num_cards)
        
        # 4) Action head: binary play vs no-play decision
        action_logits = self.action_head_mlp(shared_input)  # (B, 1)
        
        # 5) Card head: which card to play
        card_logits = self.card_head_mlp(shared_input)  # (B, num_cards)
        # apply playable mask
        card_logits[~playable_mask.bool()] = -100.0

        # 4) placement head
        # reduce channels
        x = self.place_reduce_conv(h_last)  # (B, place_mid_ch, Hf, Wf)
        # broadcast numeric_feats into spatial map
        # expand numeric_feats to (B, numeric_feat_dim, Hf, Wf)
        numeric_spatial = numeric_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Hf, Wf)
        fused = torch.cat([x, numeric_spatial], dim=1)  # (B, place_mid_ch + numeric_feat_dim, Hf, Wf)
        x = self.place_conv_block(fused)  # (B, fused_ch//2, Hf, Wf)
        
        # Output (B, num_cards, Hf, Wf)
        logits_map = self.place_logits(x) 

        # resize to grid_h x grid_w
        if (Hf, Wf) != (self.grid_h, self.grid_w):
            logits_map_resized = F.interpolate(logits_map, size=(self.grid_h, self.grid_w),
                                               mode='bilinear', align_corners=False) # (B, num_cards, grid_h, grid_w)
        else:
            logits_map_resized = logits_map  # (B, num_cards, grid_h, grid_w)

        # Flatten spatial dims: (B, num_cards, grid_h * grid_w)
        placement_logits = logits_map_resized.flatten(2)

        return {
            "action_logits": action_logits,
            "card_logits": card_logits,
            "placement_logits": placement_logits,
            "placement_map": logits_map_resized,
            "convlstm_hidden_seq": h_seq,  # (B, T, convlstm_hidden, Hf, Wf)
        }