"""
Permuted MNIST: a standard continual-learning benchmark.

Each task applies a different random pixel permutation to the original
MNIST images.  The model sees tasks sequentially and must learn all
permutations without forgetting earlier ones.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_permuted_mnist_loaders(
    num_tasks: int = 10,
    batch_size: int = 256,
    data_root: str = "./data",
    seed: int = 1004,
) -> tuple[list[DataLoader], list[DataLoader], list[DataLoader]]:
    """
    Return (train_loaders, val_loaders, test_loaders), one per task.

    Each loader yields (image, label) pairs where:
      - image shape: (batch, 784)  -- flattened and permuted
      - label shape: (batch, 10)   -- one-hot
    """
    from torchvision import datasets, transforms

    rng = np.random.RandomState(seed)

    # Download MNIST once
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    # Split original train into train (50k) and val (10k)
    all_images = train_set.data.float() / 255.0
    all_labels = train_set.targets

    train_images = all_images[:50000]
    train_labels = all_labels[:50000]
    val_images = all_images[50000:]
    val_labels = all_labels[50000:]
    test_images = test_set.data.float() / 255.0
    test_labels = test_set.targets

    # Generate random permutations (one per task)
    permutations = [rng.permutation(784) for _ in range(num_tasks)]

    train_loaders, val_loaders, test_loaders = [], [], []
    for t in range(num_tasks):
        perm = permutations[t]

        def permute(imgs: torch.Tensor, p=perm) -> torch.Tensor:
            b = imgs.view(imgs.size(0), -1)
            return b[:, p].float().clone()

        x_train = permute(train_images)
        y_train = torch.nn.functional.one_hot(train_labels, num_classes=10).float()
        x_val = permute(val_images)
        y_val = torch.nn.functional.one_hot(val_labels, num_classes=10).float()
        x_test = permute(test_images)
        y_test = torch.nn.functional.one_hot(test_labels, num_classes=10).float()

        train_set = TensorDataset(x_train, y_train)
        val_set = TensorDataset(x_val, y_val)
        test_set = TensorDataset(x_test, y_test)

        train_loaders.append(DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True))
        val_loaders.append(DataLoader(val_set, batch_size=batch_size, shuffle=False))
        test_loaders.append(DataLoader(test_set, batch_size=batch_size, shuffle=False))

    return train_loaders, val_loaders, test_loaders
