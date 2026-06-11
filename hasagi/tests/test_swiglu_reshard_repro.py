"""The certificate must catch Megatron-LM's pre-fix SwiGLU sharding bug.

End-to-end natural-bug gate: a real megatron-core gated MLP saved at TP=2
through the pre-PR#520 declaration (fused gate/up weight as one contiguous
TP-sharded tensor) and reloaded at TP=1 through the real dist_checkpointing
pipeline silently permutes gate/up rows; the transition certificate must
ABORT on exactly that arm, and COMMIT on both the same-TP control and the
shipped-fix declaration -- with zero injected faults. Requires the pinned
megatron-core venv described in ``exp_attest_swiglu_reshard_repro.py``;
skipped when it is absent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(HERE, "exp_attest_swiglu_reshard_repro.py")
MCORE_VENV = os.path.join(HERE, ".venv", "mcore")

needs_mcore_venv = pytest.mark.skipif(
    not os.path.isdir(MCORE_VENV),
    reason="needs the megatron-core==0.17.1 venv (see exp docstring)",
)


@needs_mcore_venv
def test_swiglu_missplit_caught_blind(tmp_path) -> None:
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
    arms = report["arms"]

    buggy = arms["buggy_tp2_to_tp1"]
    assert not buggy["committed"]
    assert buggy["violations_by_invariant"] == ["content_equivalence"]
    # the silent signature: every value survives (pure permutation), and the
    # loaded weight is bit-exactly the mis-split the mechanism predicts
    assert buggy["values_preserved_as_multiset"] is True
    assert buggy["fc1_matches_predicted_missplit"] is True
    assert buggy["max_abs_dev"] > 0

    # the bug fires only on the TP-degree change: same declaration, same TP
    # restores bit-exact
    assert arms["buggy_tp2_roundtrip"]["committed"]
    assert arms["buggy_tp2_roundtrip"]["max_abs_dev"] == 0.0

    assert arms["fixed_tp2_to_tp1"]["committed"]
    assert arms["fixed_tp2_to_tp1"]["n_violations"] == 0
    assert arms["fixed_tp2_to_tp1"]["max_abs_dev"] == 0.0
