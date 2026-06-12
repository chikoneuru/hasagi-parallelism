"""The auxiliary-stream residency invariant.

A wrong-but-internally-valid RNG stream is invisible to the parameter,
optimizer, progress, and reduction-order invariants; these tests pin the
sixth invariant that sees it: streams keep their logical owner and exact
content across a transition unless a re-seeding is declared.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from attest.gate import certify_transition
from attest.snapshot import fingerprint_stream, snapshot_from_state_dicts

MODEL = {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3)}


def _streams(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return {
        "ep0.random_rng_state": random.getstate(),
        "ep0.np_rng_state": np.random.get_state(),
        "ep0.torch_rng_state": torch.get_rng_state(),
    }


def test_intact_streams_commit() -> None:
    pre = snapshot_from_state_dicts(MODEL, aux_streams=_streams(7))
    post = snapshot_from_state_dicts(MODEL, aux_streams=_streams(7))
    assert certify_transition(pre, post).committed


def test_wrong_stream_restored_aborts_and_only_aux_fires() -> None:
    pre = snapshot_from_state_dicts(MODEL, aux_streams=_streams(7))
    post = snapshot_from_state_dicts(MODEL, aux_streams=_streams(8))
    decision = certify_transition(pre, post)
    assert decision.aborted
    assert {v.invariant for v in decision.violations} == {"aux_stream_residency"}
    assert all("without a declared reseed" in v.detail for v in decision.violations)


def test_lost_and_invented_streams_abort() -> None:
    streams = _streams(7)
    pre = snapshot_from_state_dicts(MODEL, aux_streams=streams)
    renamed = {("ep1." + k.split(".", 1)[1]): v for k, v in streams.items()}
    decision = certify_transition(
        pre, snapshot_from_state_dicts(MODEL, aux_streams=renamed))
    assert decision.aborted
    details = {v.detail for v in decision.violations}
    assert "stream lost in transition" in details
    assert "stream appeared from nowhere" in details


def test_declared_reseed_is_exempt() -> None:
    pre = snapshot_from_state_dicts(MODEL, aux_streams=_streams(7))
    post = snapshot_from_state_dicts(
        MODEL, aux_streams=_streams(8),
        declared_reseeds=["ep0.random_rng_state", "ep0.np_rng_state",
                          "ep0.torch_rng_state"])
    assert certify_transition(pre, post).committed


def test_reseed_declaration_does_not_waive_residency() -> None:
    pre = snapshot_from_state_dicts(MODEL, aux_streams=_streams(7))
    post = snapshot_from_state_dicts(
        MODEL, aux_streams={}, declared_reseeds=["ep0.torch_rng_state"])
    decision = certify_transition(pre, post)
    assert decision.aborted  # reseeding explains new content, not absence


def test_fingerprint_stream_is_type_aware() -> None:
    t = torch.ones(4)
    assert fingerprint_stream(t) == fingerprint_stream(t.clone())
    assert fingerprint_stream(b"abc") != fingerprint_stream("abc")
    assert fingerprint_stream((1, 2)) == fingerprint_stream((1, 2))
    assert fingerprint_stream((1, 2)) != fingerprint_stream((2, 1))
