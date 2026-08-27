"""
Text / language-modeling dataset for DEN.

Frames "teaching English" as a continual-learning problem:

  * The corpus (one story per line, e.g. TinyStories) is split into
    ``num_tasks`` contiguous chunks.  Each chunk is a *task*: a new
    slice of English the model has to learn without forgetting the
    previous ones.
  * Inside a task the model does **next-token prediction**: given the
    last ``context`` words (an n-gram window), predict the next word.
    This is a classification problem over the vocabulary, which is
    exactly what DEN's task-specific output heads support.

Samples returned by the loaders
-------------------------------
``x`` : LongTensor (B, context)   -- token ids of the context window
``y`` : FloatTensor (B, vocab)    -- one-hot over the vocabulary

``x`` is raw token ids; the learned embedding (``nn.Embedding``) lives
in the model (see ``DEN.embedder``) so it is shared across all tasks.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

UNK = "<unk>"
PAD = "<pad>"

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize_line(line: str) -> list[str]:
    return _WORD_RE.findall(line.lower())


def _build_vocab(stories: list[str], vocab_size: int) -> dict[str, int]:
    """Top-``vocab_size - 2`` words + <unk>/<pad>."""
    counter: Counter = Counter()
    for story in stories:
        counter.update(_tokenize_line(story))
    words = [w for w, _ in counter.most_common(vocab_size - 2)]
    vocab = {w: i + 2 for i, w in enumerate(words)}
    vocab[UNK] = 0
    vocab[PAD] = 1
    return vocab


class _WindowDataset(Dataset):
    """Sliding n-gram windows over a task's token stream.

    Each story is padded with ``context`` <pad> tokens; a window of the
    previous ``context`` tokens predicts the next word.  Windows never
    cross story boundaries.
    """

    def __init__(self, stories: list[str], vocab: dict[str, int], context: int):
        self.context = context
        self.pad_id = vocab[PAD]
        self.unk_id = vocab[UNK]

        self.windows: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        for story in stories:
            ids = np.array(
                [vocab.get(w, self.unk_id) for w in _tokenize_line(story)],
                dtype=np.int64,
            )
            if ids.size <= context:
                continue
            padded = np.concatenate(
                [np.full(context, self.pad_id, dtype=np.int64), ids]
            )
            # window i (i >= context) predicts padded[i] = ids[i - context]
            idx = np.arange(context, padded.size - context)
            self.windows.append(np.stack([padded[i - context: i] for i in idx]))
            self.targets.append(ids[: padded.size - 2 * context])

        if self.windows:
            self.windows = np.concatenate(self.windows)
            self.targets = np.concatenate(self.targets)
        else:
            self.windows = np.zeros((0, context), dtype=np.int64)
            self.targets = np.zeros(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.windows[idx]).long(),
            int(self.targets[idx]),
        )


def _one_hot_collate(vocab_size: int):
    def collate(batch):
        x = torch.stack([b[0] for b in batch])
        y_idx = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return x, y_idx

    return collate


def _make_loader(
    stories: list[str],
    vocab: dict[str, int],
    context: int,
    batch_size: int,
    vocab_size: int,
    max_windows: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ds = _WindowDataset(stories, vocab, context)
    if len(ds) > max_windows:
        rng = np.random.RandomState(seed)
        keep = rng.choice(len(ds), max_windows, replace=False)
        keep.sort()
        ds.windows = ds.windows[keep]
        ds.targets = ds.targets[keep]
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        collate_fn=_one_hot_collate(vocab_size),
    )


def _load_stories(data_path: str) -> list[str]:
    """Load stories from a .txt (one story per line) or a .parquet file /
    directory of .parquet files (TinyStories HF format, ``text`` column).

    Detects Git-LFS pointer files (stub text files that reference a
    remote download) and raises a helpful error.
    """
    path = Path(data_path)

    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise ValueError(
                f"No .parquet files found in {path} (expected TinyStories HF shards)"
            )
    elif path.suffix == ".parquet":
        files = [path]
    else:
        files = []

    if files:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "Reading .parquet needs pyarrow: .venv/bin/pip install pyarrow"
            )
        stories: list[str] = []
        for f in files:
            tbl = pq.read_table(f, columns=["text"])
            stories.extend(tbl["text"].to_pylist())
        return [s.strip() for s in stories if s and s.strip()]

    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if raw and "git-lfs" in raw[0]:
        raise ValueError(
            f"{path} is a Git-LFS pointer file, not the real dataset.\n"
            "Use the .parquet files instead, e.g. data_path: ../TinyStories/data"
        )
    return [line.strip() for line in raw if line.strip()]


def get_text_loaders(
    data_path: str,
    num_tasks: int = 5,
    batch_size: int = 256,
    context: int = 4,
    emb_dim: int = 32,
    vocab_size: int = 2048,
    train_frac: float = 0.8,
    max_windows: int = 300_000,
    stories_per_task: int | None = 2000,
    seed: int = 1004,
) -> tuple[list, list, list, dict]:
    """Build one (train, val, test) loader triple per task.

    The corpus is split into ``num_tasks`` contiguous story-chunks.
    Stories inside each chunk are randomly (seeded) sampled down to
    ``stories_per_task`` (so the full 2M-story TinyStories corpus stays
    CPU-friendly) and partitioned into train / val / test.
    """
    stories = _load_stories(data_path)
    # Vocabulary from a seeded sample for speed (full corpus = hundreds
    # of millions of words); deterministic across runs so checkpoints
    # stay compatible.
    vocab_rng = np.random.RandomState(seed)
    vocab_sample = stories
    if len(stories) > 200_000:
        idx = vocab_rng.choice(len(stories), 200_000, replace=False)
        vocab_sample = [stories[i] for i in idx]
    vocab = _build_vocab(vocab_sample, vocab_size)
    vocab_size_actual = len(vocab)

    chunk_edges = np.linspace(0, len(stories), num_tasks + 1, dtype=int)
    rng = np.random.RandomState(seed)

    train_loaders, val_loaders, test_loaders = [], [], []
    for t in range(num_tasks):
        chunk = stories[chunk_edges[t]: chunk_edges[t + 1]]
        if not chunk:
            raise ValueError(f"Task {t}: empty chunk (too few stories for {num_tasks} tasks)")
        if stories_per_task is not None and len(chunk) > stories_per_task:
            keep = rng.choice(len(chunk), stories_per_task, replace=False)
            chunk = [chunk[i] for i in keep]
        n_train = max(1, int(len(chunk) * train_frac))
        n_val = max(1, (len(chunk) - n_train) // 2)
        perm = rng.permutation(len(chunk))
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]
        train_loaders.append(
            _make_loader(
                [chunk[i] for i in train_idx], vocab, context,
                batch_size, vocab_size_actual, max_windows, True, seed,
            )
        )
        val_loaders.append(
            _make_loader(
                [chunk[i] for i in val_idx], vocab, context,
                batch_size, vocab_size_actual, max_windows // 4, False, seed,
            )
        )
        test_loaders.append(
            _make_loader(
                [chunk[i] for i in test_idx], vocab, context,
                batch_size, vocab_size_actual, max_windows // 4, False, seed,
            )
        )

    meta = {
        "num_classes": vocab_size_actual,
        "input_dim": context * emb_dim,
        "context": context,
        "emb_dim": emb_dim,
        "vocab_size": vocab_size_actual,
        "vocab": vocab,
        "id_to_word": {i: w for w, i in vocab.items()},
    }
    return train_loaders, val_loaders, test_loaders, meta


def save_vocab(meta: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"id_to_word": meta["id_to_word"], "context": meta["context"]}, f)


def load_vocab(path: str) -> tuple[dict[str, str], int]:
    with open(path) as f:
        data = json.load(f)
    id_to_word = {int(k): v for k, v in data["id_to_word"].items()}
    return id_to_word, int(data["context"])


def tokenize_to_ids(prompt: str, vocab: dict[str, int]) -> list[int]:
    unk = vocab[UNK]
    return [vocab.get(w, unk) for w in _WORD_RE.findall(prompt.lower())]


def generate(
    model,
    token_ids: list[int],
    id_to_word: dict[str, int],
    context: int,
    max_len: int = 30,
    temperature: float = 1.0,
    task_id: int | None = None,
    device=None,
) -> str:
    """Autoregressive generation with a trained DEN.

    ``token_ids`` is the prompt as token ids; the model's task-specific
    time-stamped forward pass is used (last trained task by default).
    When the top token would repeat the previous one, the second-best
    token is chosen instead (breaks "the the the" loops).
    """
    import torch

    model.eval()
    ids = list(token_ids)
    pad_id = 1  # <pad>
    for _ in range(max_len):
        window = (ids[-context:] if len(ids) >= context else [pad_id] * (context - len(ids)) + ids)
        x = torch.tensor([window], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x, task_id=task_id)  # (1, vocab)
        probs = torch.softmax(logits[0] / temperature, dim=-1)
        probs[0] = 0.0  # never emit <unk>
        nxt = int(probs.argmax().item())
        if ids and nxt in ids[-2:]:
            probs[nxt] = 0.0
            nxt = int(probs.argmax().item())
        if ids and nxt in ids[-2:]:  # still repeating -> pick from the rest
            probs[nxt] = 0.0
            nxt = int(probs.argmax().item())
        ids.append(nxt)
    words = [id_to_word.get(i, id_to_word.get(0, "?")) for i in ids]
    return " ".join(words)