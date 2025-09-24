# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import io
import os
import glob
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch
import webdataset as wds
from torch.utils.data import IterableDataset
from webdataset import shardlists as wds_shardlists


class CPJump1WebDataset(IterableDataset):
    """
    CPJUMP1 WebDataset reader for 5-channel microscopy tensors (.npy inside tar shards).

    Expects shards under base_path/split/*.tar, with samples containing:
      - "npy": 5xHxW (float32) for channels 1-5 (fluorescence)
      - optional per-sample mean/std: "json" dict with keys mean/std (length 5)

    Args:
        base_path: root path (e.g., /mnt/deepcad_nfs/CPJUMP1_dataset_dinov3)
        split: one of {"train","val","test"}
        transform: DINOv3 augmentation callable taking PIL/array and returning dict of crops tensors
        predata: "direct" or "crop"; if crop, pre-slice to 300x300 tiles
        tile_size: size for crop mode
    """

    def __init__(
        self,
        *,
        root: str,
        split: str,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        predata: str = "direct",
        tile_size: int = 300,
        use_dynamic_stats: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.predata = predata
        self.tile_size = tile_size
        self.use_dynamic_stats = bool(use_dynamic_stats)

        assert split in {"train", "val", "test"}
        # Expand tar shards eagerly to avoid environment-dependent glob behavior inside WebDataset
        self.urls = sorted(glob.glob(os.path.join(root, split, "*.tar")))
        if len(self.urls) == 0:
            raise RuntimeError(
                f"No .tar shards found under {os.path.join(root, split)}; please verify path and permissions"
            )

    def _iter_samples(self):
        # Use global shard list and let WebDataset handle rank/worker splitting
        all_urls = self.urls
        if len(all_urls) == 0:
            raise RuntimeError("No .tar shards available after glob.")

        # Resampled infinite stream; explicitly set nodesplitter for multi-rank training via constructor
        ds = (
            wds.WebDataset(
                all_urls,
                resampled=True,
                shardshuffle=False,  # ignored in resampled mode; keep False to silence warnings
                empty_check=False,
                nodesplitter=wds_shardlists.split_by_node,
            )
            .shuffle(256)
            .repeat()
        )
        for sample in ds:
            # robustly find .npy payload
            if "npy" in sample:
                npy_bytes = sample["npy"]
            else:
                npy_key = next((k for k in sample.keys() if k.endswith("npy")), None)
                if npy_key is None:
                    continue
                npy_bytes = sample[npy_key]

            meta = sample.get("json", None)
            arr = np.load(io.BytesIO(npy_bytes))  # expected 8x1080x1080 float32
            img = arr[:5]  # take first 5 channels (fluorescence)
            del arr

            per_mean = None
            per_std = None
            if isinstance(meta, dict):
                per_mean = meta.get("mean", None)
                per_std = meta.get("std", None)
            yield img, per_mean, per_std

    def _tiles_from_image(self, img: np.ndarray) -> list[np.ndarray]:
        C, H, W = img.shape
        ts = self.tile_size
        tiles = []
        for y in range(0, H - ts + 1, ts):
            for x in range(0, W - ts + 1, ts):
                tiles.append(img[:, y : y + ts, x : x + ts])
        return tiles

    def __iter__(self):
        for img, per_mean, per_std in self._iter_samples():
            # torchvision transforms expect HxWxC or PIL; our pipeline's DataAugmentationDINO consumes PIL
            # Here we convert 5xHxW to HxWx5 numpy and feed to transform that should handle tensors
            if self.predata == "direct":
                # img is 5xHxW float32 already; avoid extra HWC<->CHW conversions to save memory
                image_t = torch.from_numpy(img).float()  # CxHxW tensor
                sample = (image_t, ())
                if self.transform is not None:
                    # if transform supports dynamic per-sample stats, set them
                    if self.use_dynamic_stats and hasattr(self.transform, "set_dynamic_stats") and per_mean is not None and per_std is not None:
                        self.transform.set_dynamic_stats(per_mean[:5], per_std[:5])
                    sample = self.transform(image_t)
                yield sample, ()
            else:
                for tile in self._tiles_from_image(img):
                    image_t = torch.from_numpy(tile).float()  # CxHxW tensor
                    sample = (image_t, ())
                    if self.transform is not None:
                        if self.use_dynamic_stats and hasattr(self.transform, "set_dynamic_stats") and per_mean is not None and per_std is not None:
                            self.transform.set_dynamic_stats(per_mean[:5], per_std[:5])
                        sample = self.transform(image_t)
                    yield sample, ()

    def __len__(self):
        # Length is unknown for IterableDataset; training uses infinite sampler
        raise TypeError

