# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .ade20k import ADE20K
from .coco_captions import CocoCaptions
from .image_net import ImageNet
from .image_net_22k import ImageNet22k

# CPJUMP1 custom datasets (optional import)
try:
    from .cpjump1_wds import CPJump1WebDataset
    from .cpjump1_pg_npy import CPJump1PostgresNPY
except Exception:
    CPJump1WebDataset = None
    CPJump1PostgresNPY = None
