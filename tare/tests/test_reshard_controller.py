"""CPU tests for the reshard controller + live-reconfig harness (Uplift 3 scaffold).

These pin the two gates the live-reconfiguration measurement depends on:
  1. the flatten/shard round-trip is exact for any world size (so resharding never
     perturbs the weights), and the controller preserves every parameter; and
  2. verify-before-commit aborts and leaves the model at its last verified state
     when the reassembled state does not match.
Plus the end-to-end harness gate: the loss trajectory across a DDP->FSDP reshard
tracks a no-reshard control to within tolerance.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from experiments import exp04_live_reconfig
from tare.state.reshard import (
    ReshardController,
    flatten_state,
    shard_flat,
    unflatten_state,
    unshard_flat,
)


def _toy_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_flatten_unflatten_roundtrip() -> None:
    state = _toy_model().state_dict()
    flat, manifest = flatten_state(state)
    back = unflatten_state(flat, manifest)
    assert set(back) == set(state)
    for k in state:
        assert torch.allclose(back[k], state[k].to(torch.float32))


def test_shard_unshard_roundtrip_all_world_sizes() -> None:
    flat, manifest = flatten_state(_toy_model().state_dict())
    for world in (1, 2, 3, 4, 5, 8):
        shards = shard_flat(flat, world)
        assert len(shards) == world
        # FSDP-style: every shard is the same length (padded), contiguous split.
        assert len({s.numel() for s in shards}) == 1
        rebuilt = unshard_flat(shards, manifest.total_numel)
        assert rebuilt.numel() == manifest.total_numel
        assert torch.allclose(rebuilt, flat)


def test_reshard_and_commit_preserves_every_parameter() -> None:
    model = _toy_model(seed=1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rc = ReshardController()
    rc.capture(model, from_world=1)
    cert = rc.reshard_and_commit(model, to_world=4)
    assert cert.ok
    assert cert.max_abs_diff == 0.0
    after = model.state_dict()
    for k in before:
        assert torch.allclose(after[k], before[k]), k


def test_verify_detects_a_corrupted_reassembly() -> None:
    model = _toy_model(seed=2)
    rc = ReshardController(atol=1e-6)
    rc.capture(model, from_world=1)
    good = model.state_dict()
    bad = {k: v.clone() for k, v in good.items()}
    first = next(iter(bad))
    bad[first] = bad[first] + 1.0  # perturb one parameter
    cert_good = rc.verify({k: v for k, v in good.items()}, to_world=2)
    cert_bad = rc.verify(bad, to_world=2)
    assert cert_good.ok and cert_good.max_abs_diff <= 1e-6
    assert not cert_bad.ok and cert_bad.max_abs_diff >= 1.0


def test_commit_aborts_to_last_verified_state_on_mismatch() -> None:
    """If the planned shards reassemble to a wrong state, the model is untouched."""
    model = _toy_model(seed=3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rc = ReshardController(atol=1e-6)
    rc.capture(model, from_world=1)
    # Corrupt the captured flat buffer so the reassembly diverges from _captured.
    assert rc._flat is not None
    rc._flat = rc._flat + 0.5
    cert = rc.reshard_and_commit(model, to_world=2)
    assert not cert.ok                      # verification failed
    after = model.state_dict()
    for k in before:                        # model left at last verified state
        assert torch.allclose(after[k], before[k]), k


def test_live_reconfig_harness_loss_continuity_cpu() -> None:
    args = argparse.Namespace(
        d_model=32, layers=3, phase_iters=20, batch=16, to_world=2,
        lr=0.01, seed=0, atol=1e-6, tol_ce=0.3, smoke=True, out=None,
    )
    rc_code = exp04_live_reconfig.run(args)
    assert rc_code == 0  # cert.ok AND loss-continuity within tol
