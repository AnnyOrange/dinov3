"""
CPJUMP1 embedding generator.

Reads image tensor paths from PostgreSQL, loads 8-channel .npy tensors,
selects the first 5 fluorescence channels, applies 8-bit quantization
(CellCLIP-style or Global Normalization), extracts embeddings using DINOv2/v3,
aggregates across crops, saves a per-sample .npy embedding file, and records
the path in the database table `image_embeddings`.

This script is intentionally self-contained and mirrors the overall flow of
`convert_npz_to_avg_emb.py` (referenced by the user) with configurable options.

All code comments are in English per user request.
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import psycopg2
import torch
import torch.nn.functional as F


# -----------------------------
# Database helpers
# -----------------------------


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


def ensure_table_image_embeddings(conn) -> None:
    """Create image_embeddings table if it does not exist."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS image_embeddings (
                id SERIAL PRIMARY KEY,
                embedding_path TEXT NOT NULL,
                text_id INTEGER NOT NULL
            );
            """
        )
    conn.commit()


def fetch_tensor_rows(conn, limit: Optional[int] = None) -> List[Tuple[str, int]]:
    """Return list of (tensor_path, text_id)."""
    with conn.cursor() as cur:
        if limit is None:
            cur.execute("SELECT tensor_path, text_id FROM cpjump1_dinov3_tensors;")
        else:
            cur.execute("SELECT tensor_path, text_id FROM cpjump1_dinov3_tensors LIMIT %s;", (int(limit),))
        rows = cur.fetchall()
    return [(r[0], int(r[1])) for r in rows]


def insert_embedding_record(conn, embedding_path: str, text_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO image_embeddings (embedding_path, text_id) VALUES (%s, %s);",
            (embedding_path, int(text_id)),
        )
    # commit is managed by caller (batched)


# -----------------------------
# Quantization helpers
# -----------------------------


def quantize_cellclip_style(img_5ch: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """CellCLIP-style percentile clipping per channel, then scale to [0, 255] uint8.

    Args:
        img_5ch: numpy array shape (5, H, W), dtype float32 preferred.
        low_pct: lower percentile.
        high_pct: upper percentile.
    Returns:
        uint8 array (5, H, W) with values in [0, 255].
    """
    assert img_5ch.ndim == 3 and img_5ch.shape[0] == 5
    q_img = np.empty_like(img_5ch, dtype=np.uint8)
    for c in range(5):
        chan = img_5ch[c]
        lo = np.percentile(chan, low_pct)
        hi = np.percentile(chan, high_pct)
        if hi <= lo:
            hi = lo + 1e-6
        chan = np.clip(chan, lo, hi)
        chan = (chan - lo) / (hi - lo)
        chan = np.clip(np.round(chan * 255.0), 0, 255).astype(np.uint8)
        q_img[c] = chan
    return q_img


def quantize_global_norm(img_5ch: np.ndarray, mean: np.ndarray, std: np.ndarray, clamp_z: float = 3.0) -> np.ndarray:
    """Global normalization per channel: (x-mean)/std, then map [-clamp_z, +clamp_z] -> [0, 255].

    Args:
        img_5ch: (5, H, W) float32.
        mean: (5,) float.
        std: (5,) float, values > 0.
        clamp_z: range for z-score clamping before mapping to 8-bit.
    Returns:
        uint8 array (5, H, W) in [0, 255].
    """
    assert img_5ch.ndim == 3 and img_5ch.shape[0] == 5
    mean = mean.reshape(5, 1, 1)
    std = std.reshape(5, 1, 1)
    std = np.where(std <= 1e-6, 1.0, std)
    z = (img_5ch - mean) / std
    z = np.clip(z, -clamp_z, clamp_z)
    z = (z + clamp_z) / (2 * clamp_z)  # [0,1]
    q = np.clip(np.round(z * 255.0), 0, 255).astype(np.uint8)
    return q


# -----------------------------
# Model helpers
# -----------------------------


def build_dino_from_cfg(config_file: str, pretrained_weights: Optional[str], device: torch.device):
    """Build a DINOv3 model for inference using the repo's factory.

    The config file must describe a teacher backbone (teacher-only eval).
    """
    from omegaconf import OmegaConf
    from dinov3.models import build_model_for_eval

    cfg = OmegaConf.load(config_file)
    model = build_model_for_eval(cfg, pretrained_weights)
    model = model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def extract_embedding(
    model,
    img_5ch_uint8: np.ndarray,
    n_crops: int = 4,
    img_size: int = 224,
    aggregation: str = "avg",
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """Generate crop embeddings and aggregate to a single vector.

    We resize to (img_size, img_size), run model.get_intermediate_layers(..., return_class_token=True),
    and take the class token as embedding.
    """
    assert img_5ch_uint8.ndim == 3 and img_5ch_uint8.shape[0] == 5
    H, W = img_5ch_uint8.shape[1:]
    crops: List[torch.Tensor] = []
    rng = np.random.default_rng()

    # Build crops in CHW float32 [0,1]
    for i in range(max(1, n_crops)):
        # Simple random crop (fall back to center if too small)
        crop_h = crop_w = min(H, W, img_size)
        if H > crop_h and W > crop_w:
            y = int(rng.integers(0, H - crop_h + 1))
            x = int(rng.integers(0, W - crop_w + 1))
            crop = img_5ch_uint8[:, y : y + crop_h, x : x + crop_w]
        else:
            crop = img_5ch_uint8
        t = torch.from_numpy(crop.astype(np.float32) / 255.0)  # 5xhxh
        t = t.unsqueeze(0)  # 1x5xhxw
        t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
        crops.append(t.squeeze(0))

    batch = torch.stack(crops, dim=0).to(device)  # Nx5xHxW

    # Forward: take the last layer CLS token
    outputs = model.get_intermediate_layers(batch, n=1, return_class_token=True)
    # outputs is a list with one element; each element is (patch_tokens, cls_token)
    if isinstance(outputs, list) and len(outputs) >= 1:
        last = outputs[-1]
        if isinstance(last, (tuple, list)) and len(last) >= 2:
            cls_tok = last[1]  # [N, 1, D]
            emb = cls_tok.squeeze(1)  # [N, D]
        else:
            # Fallback: treat as a single tensor already [N, D]
            emb = last
    else:
        # Unexpected structure; try to flatten
        if torch.is_tensor(outputs):
            emb = outputs
        else:
            raise RuntimeError("Unexpected get_intermediate_layers output structure")

    if aggregation == "avg":
        emb_agg = emb.mean(dim=0, keepdim=False)
    elif aggregation == "max":
        emb_agg, _ = emb.max(dim=0)
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    return emb_agg.detach().cpu().numpy().astype(np.float32)


# -----------------------------
# Main pipeline
# -----------------------------


def sha1_of_string(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="CPJUMP1 Embedding Generator")
    # DB args
    parser.add_argument("--db-host", default="172.16.0.217")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", default="cpjump1")
    parser.add_argument("--db-user", default="cpjump1_user")
    parser.add_argument("--db-password", default="cpjump1_secure_pass_2024")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for a dry-run")
    parser.add_argument("--commit-every", type=int, default=100)

    # IO args
    parser.add_argument("--output-dir", required=True, help="Directory to write per-sample embedding .npy files")

    # Quantization args
    parser.add_argument("--quant-method", choices=["cellclip", "global"], default="cellclip")
    parser.add_argument("--cellclip-low", type=float, default=1.0)
    parser.add_argument("--cellclip-high", type=float, default=99.0)
    parser.add_argument("--global-mean", type=str, default=None, help="JSON path or comma-separated 5 floats")
    parser.add_argument("--global-std", type=str, default=None, help="JSON path or comma-separated 5 floats")
    parser.add_argument("--global-clamp-z", type=float, default=3.0)

    # Model args
    parser.add_argument("--config-file", required=True, help="DINOv2/v3 config YAML (teacher eval)")
    parser.add_argument("--pretrained-weights", default=None, help="Consolidated .pth or DCP directory")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--n-crops", type=int, default=4)
    parser.add_argument("--aggregation", choices=["avg", "max"], default="avg")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # DB connect
    db_cfg = DBConfig(
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )
    conn = get_db_connection(db_cfg)
    ensure_table_image_embeddings(conn)

    rows = fetch_tensor_rows(conn, limit=args.limit)
    if len(rows) == 0:
        print("No rows found in cpjump1_dinov3_tensors.")
        return

    # Prepare quantization params
    global_mean = None
    global_std = None
    if args.quant_method == "global":
        def _parse_5floats(s: str) -> np.ndarray:
            if s is None:
                raise ValueError("Global mean/std must be provided for global quantization")
            p = Path(s)
            if p.exists():
                with open(p, "r") as f:
                    data = json.load(f)
                arr = np.array(data, dtype=np.float32)
            else:
                arr = np.array([float(x) for x in s.split(",")], dtype=np.float32)
            if arr.shape[0] != 5:
                raise ValueError("Expected 5 floats for global mean/std")
            return arr

        global_mean = _parse_5floats(args.global_mean)
        global_std = _parse_5floats(args.global_std)

    # Build model
    device = torch.device(args.device)
    model = build_dino_from_cfg(args.config_file, args.pretrained_weights, device)

    # Iterate and process
    pending = 0
    for tensor_path, text_id in rows:
        try:
            arr = np.load(tensor_path)  # expected 8xHxW float32
            if arr.ndim != 3 or arr.shape[0] < 5:
                print(f"Skip invalid tensor: {tensor_path} shape={arr.shape}")
                continue
            img5 = arr[:5].astype(np.float32)
            del arr

            if args.quant_method == "cellclip":
                q = quantize_cellclip_style(img5, low_pct=args.cellclip_low, high_pct=args.cellclip_high)
            else:
                q = quantize_global_norm(img5, mean=global_mean, std=global_std, clamp_z=args.global_clamp_z)

            emb = extract_embedding(
                model,
                q,
                n_crops=args.n_crops,
                img_size=args.img_size,
                aggregation=args.aggregation,
                device=device,
            )

            # Save per-sample npy
            sha = sha1_of_string(f"{tensor_path}|{text_id}")
            out_path = os.path.join(args.output_dir, f"emb_{sha}.npy")
            np.save(out_path, emb)

            # Insert DB record
            insert_embedding_record(conn, out_path, text_id)
            pending += 1
            if pending >= args.commit_every:
                conn.commit()
                pending = 0
        except Exception as e:
            print(f"Error processing {tensor_path}: {e}")

    if pending > 0:
        conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()


