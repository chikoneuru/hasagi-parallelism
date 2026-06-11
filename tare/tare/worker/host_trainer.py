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
    layout: str = field(default="plain", init=False)
    world: int = field(default=1, init=False)

    def _make_optimizer(self, params):
        import torch

        return torch.optim.SGD(params, lr=0.01, momentum=0.9)

    def _module_factory(self):
        """A fresh module skeleton, built under a forked RNG so mid-run
        construction never perturbs the training stream."""
        import torch

        from tare.models.zoo import build_model

        with torch.random.fork_rng(devices=[]):
            return build_model(self.model_name)

    def _build_model(self) -> None:
        import torch

        from tare.models.zoo import build_model

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = build_model(self.model_name).to(self._device)
        self._optim = self._make_optimizer(self._model.parameters())
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def _build_loader(self) -> None:
        from tare.data.datasets import build_loader

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
        """Persist model + optimiser state so a resumed job continues exactly.

        State is saved in the layout-portable forms (clean-keyed full model
        state, FQN-keyed optimiser state) so a job paused under DDP/FSDP can
        resume regardless of wrapper — resume always rebuilds plain.
        """
        import torch

        from tare.state.reshard import extract_full_state, named_optim_state

        Path(self.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": extract_full_state(self._model, self.layout),
             "optim_named": named_optim_state(self._model, self.layout, self._optim),
             "iters_done": self.iters_done,
             "layout": self.layout, "world": self.world},
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
        self.layout, self.world = "plain", 1   # the wrapper died with the model
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

        self._build_model()                 # rebuild model + optimiser (plain layout)
        # Our own trusted checkpoint (the model + optimiser state written above).
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self._model.load_state_dict(state["model"])
        if "optim_named" in state:          # layout-portable FQN-keyed form
            from tare.state.reshard import load_named_optim_state

            load_named_optim_state(self._model, "plain", self._optim, state["optim_named"])
        else:                               # checkpoint from before the rewrap support
            self._optim.load_state_dict(state["optim"])
        self.layout, self.world = "plain", 1
        self.iters_done = int(state.get("iters_done", self.iters_done))
        if self._loader is None:             # only pay the cold-dataloader cost if needed
            self._build_loader()
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        self.train_iters_count(self.warmup_iters)   # clock-ramp / first-iter warmup

    def reshard(self, to_world: int, to_layout: str | None = None):
        """Move the in-memory training state to a ``to_world``-way sharded layout.

        Driven by a control-loop ``ReshardEvent``. Uses ``ReshardController``
        (capture, verify-before-commit, commit) so the layout change preserves the
        parameters exactly or aborts to the last verified state. Returns the
        ``ReshardCertificate``.

        Without ``to_layout`` only the state transport runs on the resident
        model (the original in-memory check). With ``to_layout`` (``"ddp"`` /
        ``"fsdp"`` / ``"plain"``) the rewrap is applied here for real: the old
        wrapper is torn down, the new one stood up over the verified state, and
        the optimiser rebuilt with its state (momentum buffers) carried across.
        ``ddp``/``fsdp`` require an initialized process group; world>1 ranks each
        run this same call under torchrun.
        """
        if self._model is None:
            raise RuntimeError("model must be built (cold_init/resume) before reshard()")
        if to_layout is None:
            from tare.state.reshard import ReshardController

            rc = ReshardController()
            rc.capture(self._model, from_world=self.world)
            return rc.reshard_and_commit(self._model, to_world=to_world)

        import torch
        import torch.distributed as dist

        from tare.state.reshard import live_rewrap

        device = self._device
        if device is None:
            device = next(iter(self._model.parameters())).device
            self._device = device
        model, optim, cert = live_rewrap(
            self._model, self._optim,
            layout_from=self.layout, layout_to=to_layout, to_world=to_world,
            device=device,
            optim_factory=self._make_optimizer,
            module_factory=self._module_factory,
            from_world=self.world,
        )
        self._model, self._optim = model, optim
        if cert.ok:
            # the wrapper shards across the live process group: that group's
            # size, not the requested number, is the actuated world
            actuated = (dist.get_world_size()
                        if to_layout != "plain" and dist.is_available() and dist.is_initialized()
                        else to_world)
            self.layout, self.world = to_layout, actuated
        if device.type == "cuda":
            torch.cuda.synchronize()
        return cert

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
