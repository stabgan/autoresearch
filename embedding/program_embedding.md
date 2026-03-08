# autoresearch — embedding mode

Autonomous experimentation for embedding model finetuning.

## Setup

1. **Agree on a run tag**: e.g. `emb-mar8`. Branch: `autoresearch/<tag>`.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `embedding/prepare_embedding.py` — fixed constants, data loading, evaluation. Do not modify.
   - `embedding/train_embedding.py` — the file you modify. Triplet building, training loop, hyperparameters.
4. **Verify data exists**: Check that `~/.cache/autoresearch_embedding/data/` has train.csv and test.csv symlinks. If not, run: `uv run embedding/prepare_embedding.py --train-csv <path> --test-csv <path>`.
5. **Initialize results.tsv**: Create `results.tsv` with header and baseline entry.
6. **Confirm and go**.

## Experimentation

Each experiment finetunes an embedding model. Launch: `cd embedding && uv run train_embedding.py`

**What you CAN do:**
- Modify `embedding/train_embedding.py` — everything is fair game: base model, loss function, triplet building strategy, hard negative mining, hyperparameters, training loop, batch size, epochs, margin, etc.

**What you CANNOT do:**
- Modify `embedding/prepare_embedding.py`. It contains the fixed evaluation and data loading.
- Install new packages beyond what's in `pyproject.toml`.

**The goal: get the highest accuracy_at_1.** Secondary metric: MRR. Higher is better.

**Simplicity criterion**: Same as pretraining — simpler is better at equal performance.

## Output format

```
---
accuracy_at_1:    0.850000
mrr:              0.900000
n_test:           1234
training_seconds: 180.5
total_seconds:    200.3
num_triplets:     5000
base_model:       Qwen/Qwen3-Embedding-0.6B
epochs:           3
batch_size:       16
triplet_margin:   0.05
hard_mine:        False
```

Extract key metric: `grep "^accuracy_at_1:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	accuracy_at_1	mrr	status	description
a1b2c3d	0.850000	0.900000	keep	baseline
b2c3d4e	0.870000	0.920000	keep	enable hard negative mining
c3d4e5f	0.840000	0.890000	discard	increase margin to 0.2
d4e5f6g	0.000000	0.000000	crash	OOM with batch_size=128
```

## The experiment loop

LOOP FOREVER:

1. Check git state
2. Edit `embedding/train_embedding.py` with an experimental idea
3. git commit
4. Run: `cd embedding && uv run train_embedding.py > run.log 2>&1`
5. Read results: `grep "^accuracy_at_1:\|^mrr:" run.log`
6. If empty, crashed — `tail -n 50 run.log` for traceback
7. Record in results.tsv
8. If accuracy_at_1 improved, keep. Otherwise git reset.

**NEVER STOP.** You are autonomous.

## Experiment ideas

- **Loss functions**: TripletLoss margin, CosineSimilarityLoss, MultipleNegativesRankingLoss, ContrastiveLoss
- **Hard negatives**: toggle hard mining, vary max negatives per anchor, mining strategies
- **Data**: signature separator, field ordering, quality filters (require description, min length, dedup)
- **Training**: learning rate, warmup, epochs, batch size, gradient accumulation
- **Model**: try different base models if available (but Qwen 0.6B is the default)
- **Query instruction**: Qwen-style `Instruct: ...\nQuery: ...` prefix
- **Triplet construction**: different sampling strategies, positive pair selection, negative difficulty curriculum

## Common failure modes

- OOM: reduce batch size
- No triplets: data filtering too aggressive, or min_samples_per_pid too high
- Low accuracy: margin too large (pushes everything apart), or too few epochs
- Overfitting: too many epochs on small data, try reducing epochs or adding regularization
