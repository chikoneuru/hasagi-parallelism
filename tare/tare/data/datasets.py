"""Dataset → DataLoader factory. Falls back to synthetic data if torchvision can't download."""
from __future__ import annotations

import os


def build_loader(name: str, batch_size: int = 32):
    """Return a ``DataLoader`` for ``name``. Imports torch lazily."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        try:
            from torchvision import datasets, transforms
            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
            ])
            root = os.environ.get("HASAGI_DATA_DIR", "./data_cache")
            ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
            return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
        except Exception:  # pragma: no cover
            pass
    # Synthetic fallback: random images sized like CIFAR-10.
    x = torch.randn(128, 3, 32, 32)
    y = torch.randint(0, 10, (128,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
