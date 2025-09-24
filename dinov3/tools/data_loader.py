"""
Dataset and utilities to load per-sample embedding .npy files and associated prompts from PostgreSQL.

All code comments are in English per user request.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import psycopg2
import torch
from torch.utils.data import Dataset


@dataclass
class DBConfig:
    host: str
    port: int
    database: str
    user: str
    password: str


def get_db_connection(cfg: DBConfig):
    return psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.database,
        user=cfg.user,
        password=cfg.password,
    )


def fetch_embeddings_and_texts(conn) -> List[Tuple[str, int, str]]:
    """Return list of (embedding_path, text_id, text_content)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.embedding_path, e.text_id, t.text
            FROM image_embeddings e
            JOIN texts t ON t.text_id = e.text_id;
            """
        )
        rows = cur.fetchall()
    return [(r[0], int(r[1]), r[2]) for r in rows]


class EmbeddingTextDataset(Dataset):
    def __init__(self, records: List[Tuple[str, int, str]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        path, text_id, text = self.records[idx]
        emb = np.load(path).astype(np.float32)  # (D,) or (1,D)
        emb_t = torch.from_numpy(emb).float()
        if emb_t.ndim == 1:
            pass
        elif emb_t.ndim == 2 and emb_t.shape[0] == 1:
            emb_t = emb_t[0]
        else:
            emb_t = emb_t.flatten()
        return emb_t, text


