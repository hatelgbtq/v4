"""
Split MNIST: a classic continual-learning benchmark.

The 10 MNIST classes are split into 5 binary-classification tasks:
  Task 0: digits 0,1
  Task 1: digits 2,3
  Task 2: digits 4,5
  Task 3: digits 6,7
  Task 4: digits 8,9

Each task is a 2-way classification problem.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_split_mnist_loaders(
    batch_size: int = 256,
    data_root: str = "./data",
) -> tuple[list[DataLoader], list[DataLoader], list[DataLoader]]:
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    all_images = train_set.data.float() / 255.0
    all_labels = train_set.targets

    train_images = all_images[:50000]
    train_labels = all_labels[:50000]
    val_images = all_images[50000:]
    val_labels = all_labels[50000:]
    test_images = test_set.data.float() / 255.0
    test_labels = test_set.targets

    # Group classes into tasks
    task_classes = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    train_loaders, val_loaders, test_loaders = [], [], []
    for classes in task_classes:
        # Filter samples belonging to these two classes
        for loader_list, images, labels in [
            (train_loaders, train_images, train_labels),
            (val_loaders, val_images, val_labels),
            (test_loaders, test_images, test_labels),
        ]:
            mask = (labels == classes[0]) | (labels == classes[1])
            x = images[mask].view(-1, 784).float() / 255.0
            y_labels = labels[mask]
            # Map to binary: first class → [1,0], second → [0,1]
            y = torch.zeros(y_labels.size(0), 2)
            y[y_labels == classes[0], 0] = 1.0
            y[y_labels == classes[1], 1] = 1.0
            ds = TensorDataset(x, y)
            shuffle = loader_list is train_loaders
            loader_list.append(
                DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)
            )

    return train_loaders, val_loaders, test_loaders
