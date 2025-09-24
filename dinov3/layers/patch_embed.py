# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Callable, Tuple, Union

import torch
from torch import Tensor, nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
        patch_embed_strategy: str = "standard",
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        # Strategy selection: "standard"/"inflate", "channelvit", "dichavit"
        self.strategy = patch_embed_strategy.lower()

        if self.strategy in ("standard", "inflate"):
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        elif self.strategy == "channelvit":
            # Per-channel patch embedding with attention-based fusion
            self.per_channel_proj = nn.ModuleList(
                [nn.Conv2d(1, embed_dim, kernel_size=patch_HW, stride=patch_HW) for _ in range(in_chans)]
            )
            # Channel attention to fuse per-channel embeddings
            hidden_dim = max(32, embed_dim // 16)
            self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # to compute descriptors per channel after conv
                # The input to MLP will be built manually in forward
            )
            # Learnable fusion weights per channel
            self.fuse_mlp = nn.Sequential(
                nn.Linear(in_chans, hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, in_chans, bias=True),
                nn.Sigmoid(),
            )
        elif self.strategy == "dichavit":
            # Two complementary fusion branches (dichotomous-channel attention)
            self.per_channel_proj = nn.ModuleList(
                [nn.Conv2d(1, embed_dim, kernel_size=patch_HW, stride=patch_HW) for _ in range(in_chans)]
            )
            hidden_dim = max(32, embed_dim // 16)
            # Produce two sets of channel weights to encourage diversity
            self.fuse_mlp_a = nn.Sequential(
                nn.Linear(in_chans, hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, in_chans, bias=True),
                nn.Sigmoid(),
            )
            self.fuse_mlp_b = nn.Sequential(
                nn.Linear(in_chans, hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, in_chans, bias=True),
                nn.Sigmoid(),
            )
            # Combine the two fused embeddings with learnable gates (must be 1D for FSDP)
            self.combine_gate = nn.Parameter(torch.tensor([0.5]))
        else:
            raise ValueError(f"Unknown patch_embed_strategy: {self.strategy}")

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        # patch_H, patch_W = self.patch_size
        # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        if self.strategy in ("standard", "inflate"):
            x = self.proj(x)  # B C H W
        elif self.strategy in ("channelvit", "dichavit"):
            # Apply per-channel convs then fuse
            per_ch_feats = []
            for c in range(self.in_chans):
                per_ch_feats.append(self.per_channel_proj[c](x[:, c : c + 1]))  # B, D, H', W'
            # stack as B, C_in, D, H', W'
            feats = torch.stack(per_ch_feats, dim=1)
            B, C_in, D, H, W = feats.shape
            # descriptors: mean over spatial, per channel
            ch_desc = feats.mean(dim=[2, 3, 4])  # B, C_in
            if self.strategy == "channelvit":
                weights = self.fuse_mlp(ch_desc)  # B, C_in in [0,1]
                weights = weights.view(B, C_in, 1, 1, 1)
                fused = (feats * weights).sum(dim=1)  # B, D, H, W
            else:  # dichavit
                wa = self.fuse_mlp_a(ch_desc).view(B, C_in, 1, 1, 1)
                wb = self.fuse_mlp_b(ch_desc).view(B, C_in, 1, 1, 1)
                # Encourage complementarity implicitly via separate weights
                fa = (feats * wa).sum(dim=1)  # B, D, H, W
                fb = (feats * wb).sum(dim=1)  # B, D, H, W
                gate = torch.clamp(self.combine_gate, 0.0, 1.0)
                fused = gate * fa + (1 - gate) * fb
            x = fused
        else:
            raise RuntimeError("Invalid patch embed strategy during forward")

        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        if self.strategy in ("standard", "inflate"):
            nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
        else:
            # Initialize per-channel convs
            for conv in self.per_channel_proj:
                nn.init.uniform_(conv.weight, -math.sqrt(k), math.sqrt(k))
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
            # Initialize fusion MLP(s)
            def init_mlp(mlp: nn.Sequential):
                for m in mlp:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            if self.strategy == "channelvit":
                init_mlp(self.fuse_mlp)
            else:  # dichavit
                init_mlp(self.fuse_mlp_a)
                init_mlp(self.fuse_mlp_b)

    # Adapt pretrained weights during load for compatibility across channel counts and strategies
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Handle inflate strategy: adapt proj.weight from 3->N channels
        if self.strategy in ("standard", "inflate"):
            key = prefix + "proj.weight"
            if key in state_dict:
                w = state_dict[key]
                if w.ndim == 4 and w.shape[1] == 3 and self.in_chans != 3:
                    # average over RGB and replicate to new channels
                    mean_w = w.mean(dim=1, keepdim=True)  # [out,1,kh,kw]
                    new_w = mean_w.repeat(1, self.in_chans, 1, 1)
                    # keep original RGB where applicable
                    num_copy = min(3, self.in_chans)
                    new_w[:, :num_copy] = w[:, :num_copy]
                    state_dict[key] = new_w
        else:
            # ChannelViT / DichaViT: map single proj to per-channel convs if present
            key = prefix + "proj.weight"
            if key in state_dict:
                w = state_dict.pop(key)
                bkey = prefix + "proj.bias"
                if bkey in state_dict:
                    state_dict.pop(bkey)
                # initialize per-channel convs from mean over input channel dim
                if w.ndim == 4:
                    mean_w = w.mean(dim=1, keepdim=False)  # [out, kh, kw]
                    for idx, conv in enumerate(self.per_channel_proj):
                        conv_key = prefix + f"per_channel_proj.{idx}.weight"
                        # expand to [out,1,kh,kw]
                        state_dict[conv_key] = mean_w.unsqueeze(1).to(conv.weight.dtype)
                        b = prefix + f"per_channel_proj.{idx}.bias"
                        if b in state_dict:
                            # keep existing
                            pass
        # Delegate to default loader
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
