"""
Embedding finetuning experiment script. Single-file, agent-modifiable.
Analogous to train.py — the agent modifies THIS file to improve accuracy_at_1.

Usage: uv run embedding/train_embedding.py
"""

import os
import gc
import time
import random
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from prepare_embedding import (
    TIME_BUDGET, FIELDS, MIN_FIELDS_PRESENT, DATA_DIR,
    build_signature, filter_df, norm_pid, load_data, evaluate_embedding,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EPOCHS = 3
BATCH_SIZE = 16
TRAIN_SPLIT_RATIO = 0.9
SIGNATURE_SEP = " "
TRIPLET_MARGIN = 0.05
MAX_HARD_NEGATIVES = 5
MIN_SAMPLES_PER_PID = 2
MAX_POSITIVES_PER_PID = 3
HARD_MINE = False
QUERY_INSTRUCTION = ""
RANDOM_SEED = 42
DEVICE = ""  # auto-detect if empty

# Data paths (override via env or edit directly)
TRAIN_CSV = os.getenv("TRAIN_CSV", str(DATA_DIR / "train.csv"))
TEST_CSV = os.getenv("TEST_CSV", str(DATA_DIR / "test.csv"))

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(override: str = "") -> str:
    import torch
    if override:
        return override.strip().lower()
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------------------------------------------------------------------------
# Triplet building
# ---------------------------------------------------------------------------

def build_pid_maps(df: pd.DataFrame, sep: str, max_per_pid: int, seed: int):
    """Build ppid->pids and pid->signatures mappings."""
    rng = random.Random(seed)
    ppid_to_pids = defaultdict(set)
    pid_to_sigs = defaultdict(list)

    for _, row in df.iterrows():
        ppid, pid = norm_pid(row["ppid"]), norm_pid(row["pid"])
        ppid_to_pids[ppid].add(pid)
        pid_to_sigs[pid].append(build_signature(row, sep=sep))

    ppid_to_pids = {k: list(v) for k, v in ppid_to_pids.items()}
    for pid in pid_to_sigs:
        sigs = pid_to_sigs[pid]
        if len(sigs) > max_per_pid:
            pid_to_sigs[pid] = rng.sample(sigs, max_per_pid)

    return ppid_to_pids, dict(pid_to_sigs)


def build_triplets(
    df: pd.DataFrame,
    sep: str = " ",
    min_samples: int = 2,
    max_positives: int = 3,
    max_negatives: int = 5,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """Build (anchor, positive, negative) triplets from filtered data."""
    rng = random.Random(seed)
    df = filter_df(df)
    if df.empty:
        return []

    ppid_to_pids, pid_to_sigs = build_pid_maps(df, sep, max_positives, seed)
    triplets = []

    for ppid, pids in ppid_to_pids.items():
        if len(pids) < 2:
            continue
        for pid_a in pids:
            sigs_a = pid_to_sigs.get(pid_a, [])
            if len(sigs_a) < min_samples:
                continue
            other_pids = [p for p in pids if p != pid_a]
            if not other_pids:
                continue
            for i, anchor in enumerate(sigs_a):
                for positive in sigs_a[i + 1:]:
                    neg_pids = rng.sample(other_pids, min(len(other_pids), max_negatives))
                    for pid_neg in neg_pids:
                        neg_sigs = pid_to_sigs.get(pid_neg, [])
                        if neg_sigs:
                            triplets.append((anchor, positive, rng.choice(neg_sigs)))

    return triplets


def build_triplets_hard_mined(
    model: SentenceTransformer,
    df: pd.DataFrame,
    sep: str = " ",
    min_samples: int = 2,
    max_positives: int = 3,
    max_negatives: int = 5,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """Build triplets with hardest negatives selected by cosine similarity."""
    rng = random.Random(seed)
    df = filter_df(df)
    if df.empty:
        return []

    ppid_to_pids, pid_to_sigs = build_pid_maps(df, sep, max_positives, seed)

    # Encode all signatures once
    all_pids, all_sigs = [], []
    for pid in sorted(pid_to_sigs):
        for sig in pid_to_sigs[pid]:
            all_pids.append(pid)
            all_sigs.append(sig)
    if not all_sigs:
        return []

    all_embs = np.asarray(model.encode(all_sigs, show_progress_bar=True), dtype=np.float32)
    if all_embs.ndim == 1:
        all_embs = all_embs.reshape(1, -1)
    all_embs = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-9)

    pid_to_sig_emb = defaultdict(list)
    for i in range(len(all_pids)):
        pid_to_sig_emb[all_pids[i]].append((all_sigs[i], all_embs[i]))

    triplets = []
    for ppid, pids in ppid_to_pids.items():
        if len(pids) < 2:
            continue
        for pid_a in pids:
            sig_embs_a = pid_to_sig_emb.get(pid_a, [])
            if len(sig_embs_a) < min_samples:
                continue
            other_pids = [p for p in pids if p != pid_a]
            if not other_pids:
                continue
            neg_sig_embs = []
            for pid_b in other_pids:
                neg_sig_embs.extend(pid_to_sig_emb.get(pid_b, []))
            if not neg_sig_embs:
                continue
            neg_sigs = [x[0] for x in neg_sig_embs]
            neg_embs = np.stack([x[1] for x in neg_sig_embs])
            k = min(max_negatives, len(neg_sigs))

            for i, (anchor_sig, anchor_emb) in enumerate(sig_embs_a):
                for positive_sig, _ in sig_embs_a[i + 1:]:
                    sims = anchor_emb @ neg_embs.T
                    hardest = np.argsort(sims)[::-1][:k]
                    for idx in hardest:
                        triplets.append((anchor_sig, positive_sig, neg_sigs[int(idx)]))

    if triplets and seed is not None:
        rng.shuffle(triplets)
    return triplets

# ---------------------------------------------------------------------------
# Train/holdout split
# ---------------------------------------------------------------------------

def split_train_holdout(df: pd.DataFrame, ratio: float = 0.9, seed: int = 42):
    df = filter_df(df)
    if df.empty:
        return df.copy(), pd.DataFrame()
    rng = random.Random(seed)
    idx = df.index.tolist()
    rng.shuffle(idx)
    n_train = max(1, int(len(idx) * ratio))
    return df.loc[idx[:n_train]].copy(), df.loc[idx[n_train:]].copy()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training():
    t_start = time.time()
    device = get_device(DEVICE)
    logger.info("Device: %s", device)
    logger.info("Base model: %s", BASE_MODEL)

    # Load data
    train_df, test_df = load_data(TRAIN_CSV, TEST_CSV)
    train_portion, holdout = split_train_holdout(train_df, TRAIN_SPLIT_RATIO, RANDOM_SEED)
    logger.info("Train: %d rows, holdout: %d rows, test: %d rows", len(train_portion), len(holdout), len(test_df))

    # Build triplets
    if HARD_MINE:
        model = SentenceTransformer(BASE_MODEL, device=device)
        triplets = build_triplets_hard_mined(
            model, train_portion, SIGNATURE_SEP,
            MIN_SAMPLES_PER_PID, MAX_POSITIVES_PER_PID, MAX_HARD_NEGATIVES, RANDOM_SEED,
        )
    else:
        triplets = build_triplets(
            train_portion, SIGNATURE_SEP,
            MIN_SAMPLES_PER_PID, MAX_POSITIVES_PER_PID, MAX_HARD_NEGATIVES, RANDOM_SEED,
        )
        model = SentenceTransformer(BASE_MODEL, device=device)

    logger.info("Built %d triplets", len(triplets))
    if not triplets:
        print("FAIL: no triplets generated")
        exit(1)

    instruction = QUERY_INSTRUCTION.strip()
    train_samples = [
        InputExample(texts=[f"Instruct: {instruction}\nQuery: {a}", p, n] if instruction else [a, p, n])
        for a, p, n in triplets
    ]
    dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.TripletLoss(
        model=model,
        triplet_margin=TRIPLET_MARGIN,
        distance_metric=losses.TripletDistanceMetric.COSINE,
    )

    # Train with time budget
    t_train_start = time.time()
    # Estimate steps we can fit in TIME_BUDGET
    total_steps = len(dataloader) * EPOCHS
    logger.info("Total steps: %d (%d batches x %d epochs)", total_steps, len(dataloader), EPOCHS)

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=EPOCHS,
        show_progress_bar=True,
    )
    training_time = time.time() - t_train_start

    # Evaluate
    logger.info("Evaluating...")
    metrics = evaluate_embedding(model, train_df, test_df, sep=SIGNATURE_SEP)

    # Summary
    t_end = time.time()
    total_time = t_end - t_start

    print("---")
    print(f"accuracy_at_1:    {metrics['accuracy_at_1']:.6f}")
    print(f"mrr:              {metrics['mrr']:.6f}")
    print(f"n_test:           {metrics['n_test']}")
    print(f"training_seconds: {training_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"num_triplets:     {len(triplets)}")
    print(f"base_model:       {BASE_MODEL}")
    print(f"epochs:           {EPOCHS}")
    print(f"batch_size:       {BATCH_SIZE}")
    print(f"triplet_margin:   {TRIPLET_MARGIN}")
    print(f"hard_mine:        {HARD_MINE}")

    return model, metrics


if __name__ == "__main__":
    run_training()
