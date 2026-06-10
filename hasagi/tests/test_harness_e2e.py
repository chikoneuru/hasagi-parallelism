"""End-to-end transition tests on real CPU process groups (gloo + DCP).

These spawn real process groups, so they are slower than unit tests; the
matrix here is the minimal pair that exercises both directions of a real
resharding load plus one fault per invariant family.
"""
from __future__ import annotations

import torch.multiprocessing as mp

import pytest

from attest.harness import TransitionSpec, run_transition


@pytest.fixture(scope="module", autouse=True)
def spawn_method():
    mp.set_start_method("spawn", force=True)


pytestmark = pytest.mark.slow


def test_clean_scale_in_commits():
    r = run_transition(
        TransitionSpec(preset="tiny", world_pre=2, world_post=1,
                       layout_pre="shard", layout_post="full", train_steps=1)
    )
    assert r.decision.committed, [str(v) for v in r.decision.violations]


def test_clean_scale_out_commits():
    r = run_transition(
        TransitionSpec(preset="tiny", world_pre=1, world_post=2,
                       layout_pre="full", layout_post="shard", train_steps=1)
    )
    assert r.decision.committed, [str(v) for v in r.decision.violations]


@pytest.mark.parametrize(
    "fault,invariant",
    [
        ("value_corruption", "content_equivalence"),
        ("dropped_fqn", "param_residency"),
        ("stale_optimizer_moment", "optimizer_accounting"),
        ("progress_reset", "progress_invariant"),
        ("reduction_order_swap", "reduction_order_bound"),
    ],
)
def test_fault_is_caught_and_attributed(fault, invariant):
    r = run_transition(
        TransitionSpec(preset="tiny", world_pre=2, world_post=1,
                       layout_pre="shard", layout_post="full",
                       train_steps=1, fault=fault)
    )
    assert r.decision.aborted, f"fault {fault} slipped through the gate"
    assert any(v.invariant == invariant for v in r.decision.violations), (
        f"expected {invariant}, got " + ",".join(sorted({v.invariant for v in r.decision.violations}))
    )
