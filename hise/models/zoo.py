"""Model factory + synthetic layer profiles for the partitioner.

The synthetic profiles let unit tests exercise ``partition_sequential`` without running real
profiling. Replace with measured profiles for paper-grade experiments.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from hise.parallel.partitioner import LayerProfile

if TYPE_CHECKING:  # pragma: no cover
    pass


def build_model(name: str):
    """Return a torchvision/transformers model. Imports torch lazily."""
    import torch.nn as nn  # noqa: F401
    name = name.lower()
    if name in {"resnet18", "resnet-18"}:
        from torchvision.models import resnet18
        return resnet18(weights=None, num_classes=10)
    if name in {"resnet50", "resnet-50"}:
        from torchvision.models import resnet50
        return resnet50(weights=None, num_classes=10)
    if name in {"vgg16", "vgg-16"}:
        from torchvision.models import vgg16
        return vgg16(weights=None, num_classes=10)
    raise ValueError(f"unknown model '{name}'")


def layer_profiles(name: str) -> list[LayerProfile]:
    """Synthetic per-layer profile — not measured, just shaped like a real model.

    Real evaluation should populate from a profiling run; this is sufficient for unit tests
    and the smoke experiment.
    """
    name = name.lower()
    if name in {"resnet18", "resnet-18"}:
        # 18 layer groups with varying FLOPs/activations.
        return [
            LayerProfile(index=i,
                         fwd_flops=1.5e8 * (1 + i * 0.05),
                         bwd_flops=3.0e8 * (1 + i * 0.05),
                         activation_bytes=int(56 * 56 * 64 * 4 * max(1, 18 - i) / 18))
            for i in range(18)
        ]
    if name in {"vgg16", "vgg-16"}:
        return [
            LayerProfile(index=i,
                         fwd_flops=4e8 * (1.1 ** (i // 4)),
                         bwd_flops=8e8 * (1.1 ** (i // 4)),
                         activation_bytes=int(112 * 112 * 64 * 4 / (1 + i // 4)))
            for i in range(16)
        ]
    raise ValueError(f"no layer profile registered for '{name}'")
