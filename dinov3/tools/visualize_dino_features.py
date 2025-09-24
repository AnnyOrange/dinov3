#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Visualize dense features of DINOv3 backbones on CPJUMP1 5-channel validation images.
- PCA projection of patch features to RGB (Figure 13 style)
- Cosine-similarity heatmap of a reference patch to all patches (Figure 6 style)

Supports both consolidated PyTorch checkpoints (.pth) and DCP sharded checkpoint directories
produced by this repo (will also work with official DINOv3 checkpoints).

Usage examples:

1) Visualize latest checkpoint in a training output dir (uses teacher weights):
   python tools/visualize_dino_features.py \
     --config-file /home/deepcad/xzj/dinov3/dinov3/configs/train/cpjump1_vitl16.yaml \
     --ckpt-root ./outputs/cpjump1_vitl16 \
     --dataset-root /mnt/deepcad_nfs/CPJUMP1_dataset_dinov3 \
     --split val --num-images 4 --image-size 512 --out ./vis

2) Visualize a specific consolidated checkpoint file (.pth or teacher_checkpoint.pth):
   python tools/visualize_dino_features.py \
     --config-file /home/deepcad/xzj/dinov3/dinov3/configs/train/cpjump1_vitl16.yaml \
     --pretrained /path/to/teacher_checkpoint.pth \
     --dataset-root /mnt/deepcad_nfs/CPJUMP1_dataset_dinov3 --split val --out ./vis

Notes:
- If your checkpoint has in_chans=3 (official DINOv3), pass --channel-map first3 to map 5->3 channels.
- The input H and W will be adjusted to be multiples of the patch size.
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import glob
import io
import tarfile
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from dinov3.configs.config import DinoV3SetupArgs, setup_config
from dinov3.models import build_model_for_eval
from dinov3.checkpointer.checkpointer import find_latest_checkpoint
from dinov3 import distributed  # enable single-process distributed for config scaling


def _load_val_images_from_cpjump1(
    root: str,
    split: str,
    num_images: int,
) -> List[np.ndarray]:
    """
    Load raw 5xHxW numpy arrays from CPJUMP1 *.tar shards WITHOUT webdataset dependency.
    Iterates over .tar files and pulls the first N .npy arrays found.
    """
    urls = sorted(glob.glob(os.path.join(root, split, "*.tar")))
    if len(urls) == 0:
        raise FileNotFoundError(f"No shards under {os.path.join(root, split)}")

    images: List[np.ndarray] = []
    for tar_path in urls:
        try:
            with tarfile.open(tar_path, mode="r|*") as tf:  # stream to reduce mem
                for member in tf:
                    if not member.isfile():
                        continue
                    name = member.name.lower()
                    if not name.endswith(".npy"):
                        continue
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    data = fobj.read()
                    arr = np.load(io.BytesIO(data))
                    img = arr[:5].astype("float32")  # first 5 channels
                    images.append(img)
                    if len(images) >= num_images:
                        return images
        except Exception:
            # skip unreadable tar
            continue
    if len(images) == 0:
        raise RuntimeError("No .npy images decoded from shards")
    return images


def _resize_pad_to_multiple_of_patch(image_t: torch.Tensor, target_size: int, patch: int) -> torch.Tensor:
    """
    Resize the longer side to target_size, keep aspect ratio, then pad to make H and W
    multiples of the patch size. image_t: (C,H,W)
    """
    C, H, W = image_t.shape
    long_side = max(H, W)
    scale = target_size / float(long_side)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    image_t = TF.resize(image_t, [new_h, new_w], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    pad_h = (patch - (new_h % patch)) % patch
    pad_w = (patch - (new_w % patch)) % patch
    if pad_h > 0 or pad_w > 0:
        image_t = TF.pad(image_t, [0, 0, pad_w, pad_h])  # left, top, right, bottom
    return image_t


def _map_channels(x: torch.Tensor, policy: str) -> torch.Tensor:
    if x.shape[0] == 3:
        return x
    if policy == "first3":
        return x[:3]
    elif policy == "mean3":
        m = x.mean(dim=0, keepdim=True)
        return torch.cat([m, m, m], dim=0)
    else:
        raise ValueError(f"Unknown channel-map policy: {policy}")


@torch.no_grad()

def _get_last_layer_patch_features(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """
    Returns patch features of the last layer as (C, h, w) float tensor.
    image: (1,C,H,W)
    """
    outputs = model.get_intermediate_layers(image, n=1, reshape=True, return_class_token=True, norm=True)
    patches, _cls = outputs[0]
    return patches[0].float().cpu()


def _pca_top3(x_hw_c: torch.Tensor) -> np.ndarray:
    x = x_hw_c - x_hw_c.mean(dim=0, keepdim=True)
    cov = x.T @ x / max(1, x.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    pcs = eigvecs[:, -3:]
    proj = x @ pcs
    proj_np = proj.numpy()
    proj_np = (proj_np - proj_np.min(axis=0, keepdims=True)) / (
        (proj_np.max(axis=0, keepdims=True) - proj_np.min(axis=0, keepdims=True) + 1e-6)
    )
    return proj_np


def _save_image_rgb(array_hwc: np.ndarray, path: Path):
    array_uint8 = np.clip(array_hwc * 255.0, 0, 255).astype("uint8")
    Image.fromarray(array_uint8).save(path)


def _save_colormap_gray(array_hw: np.ndarray, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(3, 3), dpi=200)
    plt.axis("off")
    plt.imshow(array_hw, cmap="viridis")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def _find_pretrained_from_root_or_path(ckpt_root: Optional[str], pretrained: Optional[str]) -> str:
    if pretrained is not None and pretrained != "":
        return os.path.abspath(pretrained)
    if ckpt_root is None:
        raise ValueError("Either --pretrained or --ckpt-root must be provided")
    latest = find_latest_checkpoint(Path(ckpt_root) / "ckpt")
    if latest is None:
        pth = Path(ckpt_root) / "teacher_checkpoint.pth"
        if pth.is_file():
            return str(pth)
        raise FileNotFoundError(f"No checkpoints found under {ckpt_root}")
    teacher_pth = latest / "teacher_checkpoint.pth"
    if teacher_pth.is_file():
        return str(teacher_pth)
    return str(latest)


def parse_args():
    p = argparse.ArgumentParser(description="DINOv3 dense feature visualizations")
    p.add_argument("--config-file", type=str, required=True, help="Path to training config.yaml")
    p.add_argument("--dataset-root", type=str, required=True, help="CPJUMP1 dataset root containing val/*.tar")
    p.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Dataset split")
    p.add_argument("--num-images", type=int, default=4, help="Number of images to visualize")
    p.add_argument("--image-size", type=int, default=512, help="Resize longer side to this before patch-aligned padding")
    p.add_argument("--out", type=str, required=True, help="Output directory for visualizations")
    p.add_argument("--ckpt-root", type=str, default=None, help="Training output dir containing ckpt/ subdir")
    p.add_argument("--pretrained", type=str, default=None, help="Explicit checkpoint path (.pth or DCP dir)")
    p.add_argument("--channel-map", type=str, default="first3", choices=["first3", "mean3"], help="When model expects 3 channels, how to map 5-channel inputs")
    p.add_argument("--ref-patch", type=str, default="center", help="Reference patch for similarity: 'center' or 'x,y' indices (0-based)")
    return p.parse_args()


def _build_eval_model(config_file: str, pretrained_path: str) -> nn.Module:
    """Build backbone and load weights for inference."""
    if not distributed.is_enabled():
        distributed.enable(overwrite=True, restrict_print_to_main_process=False)
    setup_args = DinoV3SetupArgs(config_file=config_file, pretrained_weights=pretrained_path)
    cfg = setup_config(setup_args, strict_cfg=False)
    model = build_model_for_eval(cfg, pretrained_path, shard_unsharded_model=False)
    return model.cuda().eval()


def _prepare_tensor(img_c_hw: np.ndarray, image_size: int, patch: int, in_chans: int, channel_map: str) -> torch.Tensor:
    t = torch.from_numpy(img_c_hw).float()
    if in_chans == 3 and t.shape[0] != 3:
        t = _map_channels(t, policy=channel_map)
    t = _resize_pad_to_multiple_of_patch(t, image_size, patch)
    return t.unsqueeze(0).cuda(non_blocking=True)


def _parse_ref_patch(ref: str, hw: Tuple[int, int]) -> Tuple[int, int]:
    if ref == "center":
        return hw[0] // 2, hw[1] // 2
    if "," in ref:
        x_str, y_str = ref.split(",")
        return int(x_str), int(y_str)
    raise ValueError("Invalid --ref-patch; use 'center' or 'x,y'")


def main():
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrained = _find_pretrained_from_root_or_path(args.ckpt_root, args.pretrained)
    print(f"Using checkpoint: {pretrained}")

    model = _build_eval_model(args.config_file, pretrained)

    assert hasattr(model, "patch_size"), "Expected ViT-like backbone"
    patch = int(model.patch_size)
    in_chans = int(model.patch_embed.in_chans)

    images = _load_val_images_from_cpjump1(args.dataset_root, args.split, args.num_images)

    for idx, arr in enumerate(images):
        img_t = _prepare_tensor(arr, args.image_size, patch, in_chans, args.channel_map)
        feats = _get_last_layer_patch_features(model, img_t)
        C, h, w = feats.shape
        x = feats.permute(1, 2, 0).reshape(h * w, C).contiguous()
        rgb = _pca_top3(x)
        rgb_img = rgb.reshape(h, w, 3)
        H, W = img_t.shape[-2:]
        rgb_up = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0)
        rgb_up = torch.nn.functional.interpolate(rgb_up, size=(H, W), mode="bilinear", align_corners=False)
        rgb_up = rgb_up.squeeze(0).permute(1, 2, 0).cpu().numpy()
        _save_image_rgb(rgb_up, out_dir / f"img{idx:02d}_pca_rgb.png")

        feats_n = torch.nn.functional.normalize(feats.view(C, -1).T, dim=1)
        rh, rw = _parse_ref_patch(args.ref_patch, (h, w))
        ref_vec = feats_n[rh * w + rw : rh * w + rw + 1]
        sim = torch.matmul(feats_n, ref_vec.T).squeeze(1)
        sim_map = sim.view(h, w).cpu().numpy()
        sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-6)
        _save_colormap_gray(sim_map, out_dir / f"img{idx:02d}_sim.png")

    print(f"Saved visualizations to {str(out_dir)}")


if __name__ == "__main__":
    main()
