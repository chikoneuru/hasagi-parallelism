"""Minimal PyTorch training loop — used by the smoke test and by worker main.

Real implementation will use ``torch.distributed.elastic`` rendezvous + pipeline parallelism
via ``torch.distributed.pipeline.sync.Pipe``. For now this is a single-process trainer that
runs N iterations on the chosen model and reports throughput.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TrainResult:
    iterations: int
    seconds: float
    throughput_iter_per_s: float
    final_loss: float


def train_loop(
    model_name: str = "resnet18",
    dataset: str = "cifar10",
    iterations: int = 50,
    batch_size: int = 32,
) -> TrainResult:
    """Run ``iterations`` of a tiny training loop.

    Imports torch lazily so unit tests don't pull it in. Will run on CPU if no GPU available.
    """
    try:
        import torch  # noqa: F401

        from hasagi.data.datasets import build_loader
        from hasagi.models.zoo import build_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch not installed — `pip install hasagi[dev]`") from exc

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    loader = build_loader(dataset, batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    started = time.time()
    loss_val = float("nan")
    it = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.detach().item())
        it += 1
        if it >= iterations:
            break
    elapsed = time.time() - started
    return TrainResult(
        iterations=it,
        seconds=elapsed,
        throughput_iter_per_s=it / max(elapsed, 1e-9),
        final_loss=loss_val,
    )
