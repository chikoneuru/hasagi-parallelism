"""The certificate must catch HF Trainer's mid-epoch resume reproducibility break.

End-to-end natural-bug gate against huggingface/transformers#39215 (closed
unfixed): a stochastic dataset whose items draw from the global torch
stream, a mid-epoch checkpoint, and a resume. The resumed run is not
bit-reproducible against the uninterrupted run, the certificate aborts on
the auxiliary-stream invariant, and the epoch-boundary resume control (early
RNG restore) both reproduces and commits under the same harness, so the
divergence is the bug, not a harness artifact. Requires the sweep venv with
transformers installed; skipped when absent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(HERE, "exp_attest_hf_resume_repro.py")
SWEEP_VENV = os.path.join(HERE, ".venv", "sweep")


def _have_transformers() -> bool:
    if not os.path.isdir(SWEEP_VENV):
        return False
    r = subprocess.run([f"{SWEEP_VENV}/bin/python", "-c", "import transformers"],
                       capture_output=True)
    return r.returncode == 0


needs_transformers = pytest.mark.skipif(
    not _have_transformers(),
    reason="needs the sweep venv with transformers (see exp docstring)",
)


@needs_transformers
def test_midepoch_resume_break_caught_with_control(tmp_path) -> None:
    out = tmp_path / "artifact.json"
    r = subprocess.run(
        [sys.executable, EXP, "--workdir", str(tmp_path / "work"), "--out", str(out)],
        env={**os.environ, "PYTHONPATH": HERE},
        capture_output=True, text=True, timeout=1200,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    report = json.loads(out.read_text())
    assert report["live_bug_demonstrated"] is True
    assert report["bug"]["injected_faults"] == 0
    scen = {s["scenario"]: s for s in report["scenarios"]}

    mid = scen["resume"]
    assert mid["reproducible"] is False
    assert not mid["certificate_committed"]
    assert mid["violations_by_invariant"] == ["aux_stream_residency"]

    ctrl = scen["resume_epoch_boundary_control"]
    assert ctrl["reproducible"] is True
    assert ctrl["certificate_committed"]
