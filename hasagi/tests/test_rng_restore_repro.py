"""The certificate must catch Megatron-LM's pre-fix RNG sharding bug.

End-to-end natural-bug gate: real expert-parallel rank streams saved through
the pre-PR#2658 declaration (one (pp, tp) object, data-parallel rank as
replica id) and reloaded through real dist_checkpointing silently hand the
secondary expert rank the primary's generator streams; the auxiliary-stream
invariant must catch exactly that, with every other invariant silent, and
the fixed declaration must round-trip bit-exact -- zero injected faults.
Requires the pinned megatron-core venv; skipped when absent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(HERE, "exp_attest_rng_restore_repro.py")
MCORE_VENV = os.path.join(HERE, ".venv", "mcore")

needs_mcore_venv = pytest.mark.skipif(
    not os.path.isdir(MCORE_VENV),
    reason="needs the megatron-core venv (see exp docstring)",
)


@needs_mcore_venv
def test_wrong_stream_restore_caught_blind(tmp_path) -> None:
    out = tmp_path / "artifact.json"
    r = subprocess.run(
        [sys.executable, EXP, "--workdir", str(tmp_path / "work"), "--out", str(out)],
        env={**os.environ, "PYTHONPATH": HERE},
        capture_output=True, text=True, timeout=900,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    report = json.loads(out.read_text())
    assert report["natural_bug_caught_blind"] is True
    assert report["bug"]["injected_faults"] == 0

    buggy = report["arms"]["buggy"]
    assert not buggy["committed"]
    assert buggy["violations_by_invariant"] == ["aux_stream_residency"]
    # the silent signature: only the secondary expert rank's streams are wrong,
    # and nothing else in the certified state moved
    assert all(s.startswith("ep1.") for s in buggy["violating_streams"])
    assert buggy["other_invariants_silent"] is True
    assert buggy["expert_ranks_draw_identical_randomness"] is True

    fixed = report["arms"]["fixed"]
    assert fixed["committed"]
    assert fixed["n_violations"] == 0
    assert fixed["expert_ranks_draw_identical_randomness"] is False
