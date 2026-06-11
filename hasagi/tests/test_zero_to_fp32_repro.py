"""The certificate must catch DeepSpeed#6791 through the real conversion script.

End-to-end natural-bug gate: real ZeRO-2 CPU training produces a real
checkpoint; the v0.16.0 ``zero_to_fp32.py`` that DeepSpeed itself copied into
that checkpoint silently corrupts the consolidated weights (issue #6791); the
transition certificate must ABORT on both v0.16.0 arms (including DEFAULT CLI
flags) and COMMIT on the v0.16.1 fixed converter — with zero injected faults
anywhere. Requires the two version-pinned venvs described in
``exp_attest_zero_to_fp32_repro.py``; skipped when they are absent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(HERE, "exp_attest_zero_to_fp32_repro.py")
BUGGY_VENV = os.path.join(HERE, ".venv", "ds0160")
FIXED_VENV = os.path.join(HERE, ".venv", "ds0161")

needs_pinned_venvs = pytest.mark.skipif(
    not (os.path.isdir(BUGGY_VENV) and os.path.isdir(FIXED_VENV)),
    reason="needs the deepspeed==0.16.0/0.16.1 venvs (see exp docstring)",
)


@needs_pinned_venvs
def test_natural_bug_caught_blind(tmp_path) -> None:
    out = tmp_path / "artifact.json"
    r = subprocess.run(
        [sys.executable, EXP, "--workdir", str(tmp_path / "work"), "--out", str(out)],
        env={**os.environ, "PYTHONPATH": HERE},
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    report = json.loads(out.read_text())
    assert report["natural_bug_caught_blind"] is True
    assert report["bug"]["injected_faults"] == 0
    arms = report["arms"]
    for name in ("buggy_sharded", "buggy_default"):
        assert not arms[name]["committed"], name
        assert "content_equivalence" in arms[name]["violations_by_invariant"], name
        assert arms[name]["mean_zero_frac"] > 0.5, name  # the silent all-zero signature
    assert arms["fixed"]["committed"]
    assert arms["fixed"]["n_violations"] == 0
    assert arms["fixed"]["max_abs_dev"] == 0.0  # bit-exact reconstruction
