"""Tests for the live layout rewrap (plain/DDP/FSDP wrapper flips).

CPU tests pin the layout-portable optimizer-state form and the
verify-before-commit/abort semantics of ``live_rewrap`` on the plain path.
The CUDA-gated tests run the real thing on one GPU: an actual DDP wrapper torn
down and FSDP stood up over the verified state (world=1 — FSDP degrades
FULL_SHARD to NO_SHARD but exercises the same wrapper/state-dict machinery as
a multi-rank run), with momentum carried across and training continuing
loss-identically.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import tare.state.reshard as reshard_mod
from tare.state.reshard import (
    ReshardController,
    extract_full_state,
    live_rewrap,
    load_named_optim_state,
    named_optim_state,
    shard_flat,
    wrap_layout,
)
from tare.worker.host_trainer import HostTrainer

_CUDA = torch.cuda.is_available() and torch.distributed.is_nccl_available()
cuda_only = pytest.mark.skipif(not _CUDA, reason="needs CUDA + NCCL")


def _toy(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def _toy_factory() -> nn.Module:
    # no seeding inside the fork: manual_seed would also reset the CUDA
    # generators, which fork_rng(devices=[]) does not restore
    with torch.random.fork_rng(devices=[]):
        return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def _sgd(params):
    return torch.optim.SGD(params, lr=0.05, momentum=0.9)


def _train_steps(model, opt, n: int, device=None) -> float:
    torch.manual_seed(7)
    loss = None
    for _ in range(n):
        x = torch.randn(4, 8, device=device)
        opt.zero_grad(set_to_none=True)
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
    return float(loss.item())


# --- CPU: primitives -------------------------------------------------------- #
def test_wrap_layout_plain_is_identity() -> None:
    m = _toy()
    assert wrap_layout(m, "plain", torch.device("cpu")) is m


def test_wrap_layout_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown layout"):
        wrap_layout(_toy(), "zero3", torch.device("cpu"))


def test_wrap_layout_ddp_requires_process_group() -> None:
    if torch.distributed.is_initialized():  # pragma: no cover - test isolation
        pytest.skip("a process group is already up")
    with pytest.raises(RuntimeError, match="process group"):
        wrap_layout(_toy(), "ddp", torch.device("cpu"))


def test_capture_state_matches_capture() -> None:
    m = _toy(seed=1)
    rc1, rc2 = ReshardController(), ReshardController()
    rc1.capture(m, from_world=3)
    rc2.capture_state(m.state_dict(), from_world=3)
    assert rc1.last_verified_state() is not None
    for k, v in rc1.last_verified_state().items():
        assert torch.equal(v, rc2.last_verified_state()[k])
    assert rc2.from_world == 3


def test_named_optim_state_roundtrip_plain() -> None:
    m = _toy(seed=2)
    opt = _sgd(m.parameters())
    _train_steps(m, opt, 2)
    named = named_optim_state(m, "plain", opt)
    assert named["state"], "momentum SGD must have state to carry"
    assert all(isinstance(k, str) for k in named["state"])

    m2 = _toy_factory()
    m2.load_state_dict(m.state_dict())
    opt2 = _sgd(m2.parameters())
    load_named_optim_state(m2, "plain", opt2, named)
    osd1, osd2 = opt.state_dict(), opt2.state_dict()
    assert list(osd1["state"]) == list(osd2["state"])
    for i in osd1["state"]:
        assert torch.equal(osd1["state"][i]["momentum_buffer"],
                           osd2["state"][i]["momentum_buffer"])


# --- CPU: live_rewrap on the plain path ------------------------------------- #
def test_live_rewrap_plain_to_plain_preserves_params_and_momentum() -> None:
    m = _toy(seed=3)
    opt = _sgd(m.parameters())
    _train_steps(m, opt, 3)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    before_mom = named_optim_state(m, "plain", opt)

    model, optim, cert = live_rewrap(
        m, opt, layout_from="plain", layout_to="plain", to_world=4,
        device=torch.device("cpu"), optim_factory=_sgd, module_factory=_toy_factory,
    )
    assert cert.ok and cert.max_abs_diff == 0.0 and cert.to_world == 4
    for k in before:
        assert torch.allclose(model.state_dict()[k], before[k]), k
    after_mom = named_optim_state(model, "plain", optim)
    for n_ in before_mom["state"]:
        assert torch.equal(before_mom["state"][n_]["momentum_buffer"],
                           after_mom["state"][n_]["momentum_buffer"])


def test_live_rewrap_transport_abort_restores_baseline() -> None:
    m = _toy(seed=4)
    opt = _sgd(m.parameters())
    before = {k: v.clone() for k, v in m.state_dict().items()}
    # an impossible tolerance forces the transport certificate to fail
    model, optim, cert = live_rewrap(
        m, opt, layout_from="plain", layout_to="plain", to_world=2,
        device=torch.device("cpu"), optim_factory=_sgd, module_factory=_toy_factory,
        atol=-1.0,
    )
    assert not cert.ok and "[transport]" in cert.note
    for k in before:
        assert torch.allclose(model.state_dict()[k], before[k]), k


# --- two-rank gloo: certificate decisions must be group-coordinated ---------- #
def _two_rank_worker(rank: int, world: int, port: int, divergent: bool, results) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world)
    try:
        torch.manual_seed(rank if divergent else 42)
        m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        opt = _sgd(m.parameters())
        model, _optim, cert = live_rewrap(
            m, opt, layout_from="plain", layout_to="ddp", to_world=world,
            device=torch.device("cpu"), optim_factory=_sgd, module_factory=_toy_factory,
        )
        results[rank] = {
            "ok": cert.ok, "note": cert.note,
            "wrapper": type(model).__name__,
        }
    finally:
        dist.destroy_process_group()


def _spawn_two_ranks(port: int, divergent: bool) -> dict:
    import torch.multiprocessing as mp

    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_two_rank_worker, args=(2, port, divergent, results),
                 nprocs=2, join=True)
        return dict(results)


def test_two_rank_identical_state_commits_on_both_ranks() -> None:
    res = _spawn_two_ranks(port=29560, divergent=False)
    assert set(res) == {0, 1}
    for r in res.values():
        assert r["ok"], r
        assert r["wrapper"] == "DistributedDataParallel"


def test_two_rank_divergent_state_aborts_on_both_ranks_without_deadlock() -> None:
    """DDP's constructor broadcasts rank 0's params, so a plain->ddp flip from
    rank-divergent state is not state-preserving on rank 1. The certificate
    must fail on EVERY rank (all-reduced decision) and both ranks must abort to
    the plain layout — a rank-local decision here deadlocks the collective."""
    res = _spawn_two_ranks(port=29559, divergent=True)
    assert set(res) == {0, 1}
    for r in res.values():
        assert not r["ok"], r
        assert r["wrapper"] == "Sequential", r
    notes = " | ".join(r["note"] for r in res.values())
    assert "[post-rewrap]" in notes and "[peer-rank failed]" in notes


# --- CUDA: the real DDP -> FSDP flip ---------------------------------------- #
@pytest.fixture(scope="module")
def nccl_pg():
    import torch.distributed as dist

    if not _CUDA:  # pragma: no cover
        pytest.skip("needs CUDA + NCCL")
    created = False
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29558",
                                rank=0, world_size=1)
        created = True
    torch.cuda.set_device(0)
    yield torch.device("cuda:0")
    if created:
        dist.destroy_process_group()


@cuda_only
def test_shard_flat_pads_on_the_flat_tensors_device() -> None:
    flat = torch.arange(10, dtype=torch.float32, device="cuda:0")
    shards = shard_flat(flat, 3)  # 10 % 3 != 0 -> exercises the padding branch
    assert len(shards) == 3
    assert all(s.device.type == "cuda" for s in shards)


@cuda_only
def test_live_rewrap_ddp_to_fsdp_and_back(nccl_pg) -> None:
    device = nccl_pg
    model = wrap_layout(_toy(seed=5).to(device), "ddp", device)
    opt = _sgd(model.parameters())
    _train_steps(model, opt, 3, device=device)
    baseline = extract_full_state(model, "ddp")
    baseline_mom = named_optim_state(model, "ddp", opt)

    model, opt, cert = live_rewrap(
        model, opt, layout_from="ddp", layout_to="fsdp", to_world=1,
        device=device, optim_factory=_sgd, module_factory=_toy_factory,
    )
    assert cert.ok and cert.max_abs_diff == 0.0
    after = extract_full_state(model, "fsdp")
    for k in baseline:
        assert torch.equal(after[k], baseline[k]), k
    after_mom = named_optim_state(model, "fsdp", opt)
    assert after_mom["state"], "momentum must survive the rewrap"
    for n_ in baseline_mom["state"]:
        assert torch.allclose(
            after_mom["state"][n_]["momentum_buffer"].cpu().float(),
            baseline_mom["state"][n_]["momentum_buffer"].cpu().float(),
        ), n_

    # training continues under FSDP, loss-identically to a plain continuation
    ref = _toy_factory().to(device)
    ref.load_state_dict(baseline)
    ref_opt = _sgd(ref.parameters())
    load_named_optim_state(ref, "plain", ref_opt, baseline_mom)
    loss_fsdp = _train_steps(model, opt, 1, device=device)
    loss_ref = _train_steps(ref, ref_opt, 1, device=device)
    assert abs(loss_fsdp - loss_ref) < 1e-6

    model, opt, cert = live_rewrap(
        model, opt, layout_from="fsdp", layout_to="ddp", to_world=1,
        device=device, optim_factory=_sgd, module_factory=_toy_factory,
    )
    assert cert.ok and cert.max_abs_diff == 0.0


@cuda_only
def test_live_rewrap_post_wrap_abort_rewraps_old_layout(nccl_pg, monkeypatch) -> None:
    from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817

    device = nccl_pg
    model = wrap_layout(_toy(seed=6).to(device), "ddp", device)
    opt = _sgd(model.parameters())
    baseline = extract_full_state(model, "ddp")

    real_extract = reshard_mod.extract_full_state

    def corrupting(m, layout):
        sd = real_extract(m, layout)
        if layout == "fsdp":  # corrupt only the post-rewrap gather
            k = next(iter(sd))
            sd[k] = sd[k] + 1.0
        return sd

    monkeypatch.setattr(reshard_mod, "extract_full_state", corrupting)
    model, opt, cert = live_rewrap(
        model, opt, layout_from="ddp", layout_to="fsdp", to_world=1,
        device=device, optim_factory=_sgd, module_factory=_toy_factory,
    )
    assert not cert.ok and "[post-rewrap]" in cert.note
    assert isinstance(model, DDP), "abort must hand back the original layout"
    restored = real_extract(model, "ddp")
    for k in baseline:
        assert torch.equal(restored[k], baseline[k]), k


@cuda_only
def test_host_trainer_live_reshard_to_fsdp_and_back(nccl_pg) -> None:
    from tare.models.zoo import build_model

    device = nccl_pg
    ht = HostTrainer()
    ht._device = device
    ht._model = build_model("resnet18").to(device)
    ht._optim = ht._make_optimizer(ht._model.parameters())
    before = extract_full_state(ht._model, "plain")

    cert = ht.reshard(to_world=1, to_layout="fsdp")
    assert cert.ok and cert.max_abs_diff == 0.0
    assert ht.layout == "fsdp" and ht.world == 1
    after = extract_full_state(ht._model, "fsdp")
    for k in before:
        assert torch.equal(after[k], before[k]), k

    cert = ht.reshard(to_world=1, to_layout="ddp")
    assert cert.ok
    assert ht.layout == "ddp"
