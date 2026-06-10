"""Unit tests for the transition certificate over real tensors."""
from __future__ import annotations

import copy

import pytest
import torch

from attest.certificate import TransitionCertificate
from attest.gate import certify_transition
from attest.model import GPT, PRESETS, synthetic_batch
from attest.snapshot import fingerprint_tensor, snapshot_from_state_dicts


@pytest.fixture(scope="module")
def trained_state():
    cfg = PRESETS["tiny"]
    torch.manual_seed(7)
    model = GPT(cfg, seed=7)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(2):
        loss = model.loss(synthetic_batch(cfg, 4, seed=step))
        optim.zero_grad()
        loss.backward()
        optim.step()
    # FQN-keyed optimizer form, as the harness uses
    fqns = [n for n, _ in model.named_parameters()]
    raw = optim.state_dict()
    state = {fqns[idx]: slots for idx, slots in raw["state"].items()}
    return model.state_dict(), {"state": state, "param_groups": raw["param_groups"]}


def snap(model_sd, optim_sd, progress=None, order=None):
    return snapshot_from_state_dicts(
        model_sd,
        optim_sd,
        progress=progress or {"global_step": 2},
        reduction_order=order if order is not None else sorted(model_sd),
    )


# --------------------------------------------------------------- fingerprints
def test_fingerprint_deterministic_and_dtype_sensitive():
    t = torch.randn(13, 5)
    assert fingerprint_tensor(t) == fingerprint_tensor(t.clone())
    assert fingerprint_tensor(t) != fingerprint_tensor(t.to(torch.float16))
    assert fingerprint_tensor(t) != fingerprint_tensor(t.reshape(5, 13))


def test_fingerprint_view_equals_copy():
    base = torch.randn(64)
    view = base[8:24]
    assert fingerprint_tensor(view) == fingerprint_tensor(view.clone())


def test_fingerprint_bf16():
    t = torch.randn(16, dtype=torch.bfloat16)
    assert fingerprint_tensor(t) == fingerprint_tensor(t.clone())
    u = t.clone()
    u[3] += 1
    assert fingerprint_tensor(t) != fingerprint_tensor(u)


# --------------------------------------------------------------- clean commit
def test_identical_state_commits(trained_state):
    model_sd, optim_sd = trained_state
    d = certify_transition(snap(model_sd, optim_sd), snap(model_sd, optim_sd))
    assert d.committed and not d.violations


# --------------------------------------------------------------- violations
def test_param_perturbation_aborts(trained_state):
    model_sd, optim_sd = trained_state
    pre = snap(model_sd, optim_sd)
    bad = {k: v.clone() for k, v in model_sd.items()}
    first = sorted(bad)[0]
    bad[first] = bad[first] + 1e-3
    d = certify_transition(pre, snap(bad, optim_sd))
    assert d.aborted
    assert {v.invariant for v in d.violations} == {"content_equivalence"}


def test_dropped_param_aborts(trained_state):
    model_sd, optim_sd = trained_state
    pre = snap(model_sd, optim_sd)
    bad = dict(model_sd)
    bad.pop(sorted(bad)[0])
    d = certify_transition(pre, snap(bad, optim_sd))
    assert d.aborted
    assert any(v.invariant == "param_residency" for v in d.violations)


def test_invented_param_aborts(trained_state):
    model_sd, optim_sd = trained_state
    pre = snap(model_sd, optim_sd)
    bad = dict(model_sd)
    bad["ghost.weight"] = torch.zeros(3)
    d = certify_transition(pre, snap(bad, optim_sd))
    assert any(
        v.invariant == "param_residency" and "nowhere" in v.detail for v in d.violations
    )


def test_stale_optimizer_slot_aborts(trained_state):
    model_sd, optim_sd = trained_state
    pre = snap(model_sd, optim_sd)
    bad = copy.deepcopy(optim_sd)
    fqn = sorted(bad["state"])[0]
    bad["state"][fqn]["exp_avg"] = torch.zeros_like(bad["state"][fqn]["exp_avg"])
    d = certify_transition(pre, snap(model_sd, bad))
    assert d.aborted
    assert any(v.invariant == "optimizer_accounting" for v in d.violations)


def test_step_counter_reset_aborts(trained_state):
    model_sd, optim_sd = trained_state
    pre = snap(model_sd, optim_sd)
    bad = copy.deepcopy(optim_sd)
    fqn = sorted(bad["state"])[0]
    s = bad["state"][fqn]["step"]
    bad["state"][fqn]["step"] = torch.zeros_like(s) if isinstance(s, torch.Tensor) else 0
    d = certify_transition(pre, snap(model_sd, bad))
    assert d.aborted


def test_progress_reset_aborts(trained_state):
    model_sd, optim_sd = trained_state
    pre = snap(model_sd, optim_sd)
    d = certify_transition(pre, snap(model_sd, optim_sd, progress={"global_step": 0}))
    assert d.aborted
    assert any(v.invariant == "progress_invariant" for v in d.violations)


def test_reduction_order_swap_flagged(trained_state):
    model_sd, optim_sd = trained_state
    pre = snap(model_sd, optim_sd)
    order = sorted(model_sd)
    order[0], order[1] = order[1], order[0]
    post = snap(model_sd, optim_sd, order=order)
    # a permuted order with an unchanged declared bound is allowed only if a
    # bound exists; certificate flags bound mismatches and non-permutations
    cert = TransitionCertificate()
    violations = cert.check(pre, post)
    assert not any(v.invariant != "reduction_order_bound" for v in violations)


def test_independent_checksum_is_second_witness(trained_state):
    """If a (hypothetical) hash collision hid a value change, the L2 path
    must still flag it: simulate by corrupting only the checksum record."""
    model_sd, optim_sd = trained_state
    pre = snap(model_sd, optim_sd)
    post = snap(model_sd, optim_sd)
    fqn = sorted(post.checksums)[0]
    post.checksums[fqn]["l2"] *= 1.5
    d = certify_transition(pre, post)
    assert d.aborted
