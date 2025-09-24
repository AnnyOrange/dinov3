# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import io
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import psycopg2
from torch.utils.data import Dataset


@dataclass
class PostgresConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str


class CPJump1PostgresNPY(Dataset):
    """
    Debug/sanity dataset that reads metadata from PostgreSQL and loads .npy files directly.
    Only channels 1-5 are used.

    Table schema expectation (cpjump1_dinov3_tensors):
      - id (PK)
      - tensor_path (text): absolute path to .npy file (8x1080x1080 float32)
      - mean (float[8])
      - std (float[8])
      - text_id (integer): used as classification label

    Args:
        root: unused; kept for signature consistency
        split: one of {"train","val","test"} used to filter by split column
        transform: augmentation function
        conn: override connection config
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
        pg_host: str = "172.16.0.217",
        pg_port: int = 5432,
        pg_db: str = "cpjump1",
        pg_user: str = "cpjump1_user",
        pg_password: str = "cpjump1_secure_pass_2024",
    ) -> None:
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.predata = predata
        self.tile_size = tile_size
        self.use_dynamic_stats = bool(use_dynamic_stats)

        self.conn_info = PostgresConfig(
            host=pg_host, port=pg_port, dbname=pg_db, user=pg_user, password=pg_password
        )
        assert split in {"train", "val", "test"}
        self.split = split
        self.entries = self._load_entries()

    def _load_entries(self):
        conn = psycopg2.connect(
            host=self.conn_info.host,
            port=self.conn_info.port,
            dbname=self.conn_info.dbname,
            user=self.conn_info.user,
            password=self.conn_info.password,
        )
        cur = conn.cursor()
        cur.execute(
            """
            SELECT tensor_path, mean, std, text_id
            FROM cpjump1_dinov3_tensors
            WHERE split = %s
            """,
            (self.split,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows

    def __len__(self) -> int:
        return len(self.entries)

    def _tiles_from_image(self, img: np.ndarray) -> list[np.ndarray]:
        C, H, W = img.shape
        ts = self.tile_size
        tiles = []
        for y in range(0, H - ts + 1, ts):
            for x in range(0, W - ts + 1, ts):
                tiles.append(img[:, y : y + ts, x : x + ts])
        return tiles

    def __getitem__(self, index: int):
        tensor_path, mean, std, text_id = self.entries[index]
        arr = np.load(tensor_path)  # 8x1080x1080 float32
        img = arr[:5]
        if self.predata == "direct":
            image = np.moveaxis(img, 0, -1)
            if self.transform is not None and self.use_dynamic_stats and hasattr(self.transform, "set_dynamic_stats") and mean is not None and std is not None:
                self.transform.set_dynamic_stats(mean[:5], std[:5])
            sample = self.transform(image) if self.transform is not None else (image, ())
            return sample, text_id
        else:
            # Return first tile by default for debugging; production should wrap with sampler that iterates tiles
            tiles = self._tiles_from_image(img)
            image = np.moveaxis(tiles[0], 0, -1)
            if self.transform is not None and self.use_dynamic_stats and hasattr(self.transform, "set_dynamic_stats") and mean is not None and std is not None:
                self.transform.set_dynamic_stats(mean[:5], std[:5])
            sample = self.transform(image) if self.transform is not None else (image, ())
            return sample, text_id

