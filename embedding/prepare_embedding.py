"""
Fixed constants, data utilities, and evaluation for embedding experiments.
Analogous to prepare.py for the pretraining loop — this file is NOT modified by the agent.

Usage (one-time): uv run embedding/prepare_embedding.py --train-csv data/train.csv --test-csv data/test.csv
"""

import os
import math
import logging
import argparse
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300  # training time budget in seconds (5 minutes)
FIELDS = ["group_title", "episode_title", "description"]
MIN_FIELDS_PRESENT = 2

# Paths
CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "autoresearch_embedding"
DATA_DIR = CACHE_DIR / "data"

# ---------------------------------------------------------------------------
# Signature utilities
# ---------------------------------------------------------------------------

def _is_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    return not str(val).strip()


def count_present_fields(row: pd.Series, fields: list = FIELDS) -> int:
    return sum(1 for f in fields if f in row.index and not _is_empty(row.get(f)))


def build_signature(row: pd.Series, sep: str = " ", fields: list = FIELDS) -> str:
    parts = []
    for f in fields:
        val = row.get(f)
        parts.append("" if _is_empty(val) else str(val).strip())
    return sep.join(parts)


def filter_df(df: pd.DataFrame, n: int = MIN_FIELDS_PRESENT) -> pd.DataFrame:
    mask = df.apply(lambda row: count_present_fields(row) >= n, axis=1)
    return df.loc[mask].copy()


def norm_pid(pid) -> int:
    if hasattr(pid, "item"):
        return int(pid.item())
    return int(pid)

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_embedding(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sep: str = " ",
    top_k: int = 10,
) -> dict:
    """
    Fixed evaluation: build reference index from train_df, retrieve on test_df.
    Returns dict with accuracy_at_1, mrr, n_test.
    Model must have an .encode(texts, ...) method returning numpy arrays.
    """
    train_df = filter_df(train_df)
    test_df = filter_df(test_df)
    if train_df.empty or test_df.empty:
        return {"accuracy_at_1": 0.0, "mrr": 0.0, "n_test": 0}

    train_sigs = [build_signature(row, sep=sep) for _, row in train_df.iterrows()]
    train_pids = [norm_pid(row["pid"]) for _, row in train_df.iterrows()]
    test_sigs = [build_signature(row, sep=sep) for _, row in test_df.iterrows()]
    test_pids = [norm_pid(row["pid"]) for _, row in test_df.iterrows()]

    train_emb = np.asarray(model.encode(train_sigs, show_progress_bar=False), dtype=np.float32)
    test_emb = np.asarray(model.encode(test_sigs, show_progress_bar=False), dtype=np.float32)

    if train_emb.ndim == 1:
        train_emb = train_emb.reshape(1, -1)
    if test_emb.ndim == 1:
        test_emb = test_emb.reshape(1, -1)

    # L2 normalize
    train_emb = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-9)
    test_emb = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-9)

    sim = test_emb @ train_emb.T

    correct_at_1 = 0
    reciprocal_ranks = []

    for i in range(len(test_pids)):
        true_pid = test_pids[i]
        order = np.argsort(sim[i])[::-1][:top_k]
        pred_pids = [train_pids[j] for j in order]
        if pred_pids and pred_pids[0] == true_pid:
            correct_at_1 += 1
        for r, pid in enumerate(pred_pids, start=1):
            if pid == true_pid:
                reciprocal_ranks.append(1.0 / r)
                break

    n = len(test_pids)
    return {
        "accuracy_at_1": correct_at_1 / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "n_test": n,
    }

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(train_csv: str, test_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize train/test CSVs."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    for df in [train_df, test_df]:
        df.columns = df.columns.str.strip()
        for col in ["ppid", "pid"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    return train_df, test_df

# ---------------------------------------------------------------------------
# Main: validate data exists
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Prepare data for embedding experiments")
    parser.add_argument("--train-csv", required=True, help="Path to train CSV")
    parser.add_argument("--test-csv", required=True, help="Path to test CSV")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for path, name in [(args.train_csv, "train"), (args.test_csv, "test")]:
        if not Path(path).is_file():
            logger.error("%s CSV not found: %s", name, path)
            exit(1)
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        filtered = filter_df(df)
        logger.info("%s: %d rows, %d after filtering (>=%d fields)", name, len(df), len(filtered), MIN_FIELDS_PRESENT)

    # Symlink data into cache for convenience
    for src, dst_name in [(args.train_csv, "train.csv"), (args.test_csv, "test.csv")]:
        dst = DATA_DIR / dst_name
        src_abs = Path(src).resolve()
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src_abs)
        logger.info("Linked %s -> %s", dst, src_abs)

    print("\nDone! Data ready for embedding experiments.")
