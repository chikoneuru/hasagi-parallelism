"""A real GPU training job that can checkpoint, release the GPU, and resume.

This is the host-side training process gated by a serverless pod's lifecycle:
the pod's scale signal decides *when* to run, but the model, optimiser, and CUDA
context live here on the GPU. A pause must therefore pay a real resume cost on
the way back — checkpoint write/read, optimiser-state reload, CUDA
re-initialisation, and first-iteration warmup — which is exactly the
training-specific cost a stateless serverless function never incurs.

torch is imported lazily inside the methods so this module imports cleanly
without torch or a GPU present (e.g. during test collection).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HostTrainer:
    """A real resnet18/CIFAR-10 (by default) training job with pause/resume.

    Args:
        model_name / dataset / batch_size: workload definition.
        ckpt_path: where pause writes and resume reads model + optimiser state.
        warmup_iters: iterations run right after a (re)build to warm the kernels;
            counted as part of the cold-start / resume cost.
    """

    model_name: str = "resnet18"
    dataset: str = "cifar10"
    batch_size: int = 32
    ckpt_path: str = "./artifacts/host_trainer_ckpt.pt"
    warmup_iters: int = 2

    _model: object = field(default=None, init=False, repr=False)
    _optim: object = field(default=None, init=False, repr=False)
    _loader_iter: object = field(default=None, init=False, repr=False)
    _loader: object = field(default=None, init=False, repr=False)
    _device: object = field(default=None, init=False, repr=False)
    _loss_fn: object = field(default=None, init=False, repr=False)
    iters_done: int = field(default=0, init=False)

    def _build_model(self) -> None:
        import torch

        from hasagi.models.zoo import build_model

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = build_model(self.model_name).to(self._device)
        self._optim = torch.optim.SGD(self._model.parameters(), lr=0.01, momentum=0.9)
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def _build_loader(self) -> None:
        from hasagi.data.datasets import build_loader

        self._loader = build_loader(self.dataset, batch_size=self.batch_size)
        self._loader_iter = iter(self._loader)

    def _build(self) -> None:
        self._build_model()
        self._build_loader()

    def _next_batch(self):
        try:
            return next(self._loader_iter)
        except StopIteration:
            self._loader_iter = iter(self._loader)
            return next(self._loader_iter)

    def cold_init(self) -> None:
        """First-ever start: build the model and force the CUDA context up."""
        import torch

        self._build()
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        self.train_iters_count(self.warmup_iters)   # warm the kernels

    def checkpoint(self) -> None:
        """Persist model + optimiser state so a resumed job continues exactly."""
        import torch

        Path(self.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": self._model.state_dict(), "optim": self._optim.state_dict(),
             "iters_done": self.iters_done},
            self.ckpt_path,
        )
        if self._device.type == "cuda":
            torch.cuda.synchronize()

    def teardown(self, keep_dataloader: bool = False) -> None:
        """Release the GPU as scale-to-zero would: drop the model + free memory.

        ``keep_dataloader=True`` leaves the dataloader (and its worker processes)
        alive. On a host whose process survives the pause, this avoids paying the
        cold-dataloader first-batch cost on resume — a measured ~0.76 s that
        otherwise dominates the discrete resume cost. The GPU-resident state
        (model, optimiser) is always freed.
        """
        import torch

        self._model = None
        self._optim = None
        if not keep_dataloader:
            self._loader = None
            self._loader_iter = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def resume(self) -> None:
        """Real resume cost: rebuild model, reload state, warm up. Rebuilds the
        dataloader only if it was torn down (see ``teardown(keep_dataloader)``)."""
        import torch

        self._build_model()                 # rebuild model + optimiser
        # Our own trusted checkpoint (the model + optimiser state written above).
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self._model.load_state_dict(state["model"])
        self._optim.load_state_dict(state["optim"])
        self.iters_done = int(state.get("iters_done", self.iters_done))
        if self._loader is None:             # only pay the cold-dataloader cost if needed
            self._build_loader()
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        self.train_iters_count(self.warmup_iters)   # clock-ramp / first-iter warmup

    def train_iters_count(self, n: int) -> int:
        """Run exactly ``n`` real training iterations on the GPU."""
        import torch

        self._model.train()
        done = 0
        for _ in range(n):
            inputs, targets = self._next_batch()
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            self._optim.zero_grad()
            out = self._model(inputs)
            loss = self._loss_fn(out, targets)
            loss.backward()
            self._optim.step()
            done += 1
            self.iters_done += 1
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        return done

    def train_for(self, seconds: float) -> int:
        """Train for at least ``seconds`` of wall-clock; return iterations run."""
        start = time.monotonic()
        done = 0
        while time.monotonic() - start < seconds:
            done += self.train_iters_count(4)
        return done
