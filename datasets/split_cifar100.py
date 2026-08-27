"""
Split CIFAR-100: a real-world continual-learning benchmark.

The 100 fine-grained classes are split into sequential tasks:
  - 5 tasks  x 20 classes (default)
  - 10 tasks x 10 classes

Each task is a multi-way classification problem over its own class
subset.  Images are flattened to 3072-d (32x32x3) inputs, matching
DEN's MLP architecture.

Data is loaded from the pickle files (no torchvision required):
``data/cifar-100-python/{train,test,meta}``
"""

import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def _ensure_extracted(data_root: str = "./data") -> Path:
    root = Path(data_root) / "cifar-100-python"
    if (root / "train").exists():
        return root

    tarball = Path(data_root) / "cifar-100-python.tar.gz"
    if not tarball.exists():
        import urllib.request
        tarball.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [*] Downloading CIFAR-100 from {URL} ...")
        try:
            urllib.request.urlretrieve(URL, tarball)
        except Exception as e:
            tarball.unlink(missing_ok=True)
            raise RuntimeError(
                f"Download failed ({e}). Use a faster mirror:\n"
                "  aria2c -x 8 -s 8 https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            ) from e
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(data_root)
    tarball.unlink(missing_ok=True)
    return root


def _load_data(data_root: str = "./data"):
    root = _ensure_extracted(data_root)

    def unpickle(name: str):
        with open(root / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d

    train = unpickle("train")
    test = unpickle("test")

    x_tr = train[b"data"]
    x_te = test[b"data"]

    ch_mean = x_tr.reshape(-1, 3, 1024).mean(axis=(0, 2)) / 255.0
    ch_std = x_tr.reshape(-1, 3, 1024).std(axis=(0, 2)) / 255.0
    ch_mean = np.repeat(ch_mean, 1024).astype(np.float32)
    ch_std = np.repeat(ch_std, 1024).astype(np.float32)
    ch_std[ch_std < 1e-8] = 1.0

    return (
        x_tr,
        np.asarray(train[b"fine_labels"]),
        x_te,
        np.asarray(test[b"fine_labels"]),
        ch_mean,
        ch_std,
    )


def get_split_cifar100_loaders(
    num_tasks: int = 5,
    batch_size: int = 256,
    data_root: str = "./data",
    seed: int = 1004,
) -> tuple[list[DataLoader], list[DataLoader], list[DataLoader]]:
    if 100 % num_tasks != 0:
        raise ValueError("num_tasks must divide 100 evenly (5 or 10)")

    X_tr, y_tr, X_te, y_te, ch_mean, ch_std = _load_data(data_root)
    classes_per_task = 100 // num_tasks

    rng = np.random.RandomState(seed)
    train_loaders, val_loaders, test_loaders = [], [], []

    for t in range(num_tasks):
        cls = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        tr_mask = np.isin(y_tr, cls)
        te_mask = np.isin(y_te, cls)

        x_tr = (X_tr[tr_mask].astype(np.float32) / 255.0 - ch_mean) / ch_std
        y_tr_t = y_tr[tr_mask] - t * classes_per_task

        x_te = (X_te[te_mask].astype(np.float32) / 255.0 - ch_mean) / ch_std
        y_te_t = y_te[te_mask] - t * classes_per_task

        perm = rng.permutation(len(x_tr))
        n_val = max(1, int(len(x_tr) * 0.1))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        def one_hot(y: np.ndarray) -> torch.Tensor:
            return torch.nn.functional.one_hot(
                torch.from_numpy(y), num_classes=classes_per_task
            ).float()

        def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool):
            ds = TensorDataset(
                torch.from_numpy(np.ascontiguousarray(x)),
                one_hot(y),
            )
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=shuffle,
            )

        train_loaders.append(make_loader(x_tr[train_idx], y_tr_t[train_idx], True))
        val_loaders.append(make_loader(x_tr[val_idx], y_tr_t[val_idx], False))
        test_loaders.append(make_loader(x_te, y_te_t, False))

    return train_loaders, val_loaders, test_loaders