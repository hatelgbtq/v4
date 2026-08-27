"""
Standard MNIST loader (single task).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def get_mnist_loaders(
    batch_size: int = 256,
    data_root: str = "./data",
) -> DataLoader:
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    # Build loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
