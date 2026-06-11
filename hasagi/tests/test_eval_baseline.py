"""The one-batch-eval baseline must be priced honestly against the certificate.

Pins the three facts the comparison rests on: the strongest eval oracle DOES
catch both reproduced natural bugs (the honest concession), it is structurally
blind (output deviation exactly 0.0) to optimizer/progress corruption that the
certificate aborts on, and it misses near-tolerance value damage at realistic
thresholds. Requires all three pinned venvs (the natural-case states are
regenerated through the real pipelines); skipped when absent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(HERE, "exp_attest_eval_baseline.py")
VENVS = [os.path.join(HERE, ".venv", v) for v in ("ds0160", "ds0161", "mcore")]

needs_pinned_venvs = pytest.mark.skipif(
    not all(os.path.isdir(v) for v in VENVS),
    reason="needs the deepspeed and megatron-core venvs (see exp docstrings)",
)


@needs_pinned_venvs
def test_eval_baseline_priced_against_certificate(tmp_path) -> None:
    out = tmp_path / "artifact.json"
    r = subprocess.run(
        [sys.executable, EXP, "--out", str(out),
         "--ds-workdir", str(tmp_path / "ds"), "--sw-workdir", str(tmp_path / "sw")],
        env={**os.environ, "PYTHONPATH": HERE},
        capture_output=True, text=True, timeout=1500,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    report = json.loads(out.read_text())
    h = report["headline"]
    cases = {c["name"]: c for c in report["cases"]}

    # the honest concession: catastrophic natural corruption trips even a
    # one-batch eval, provided a bit-exact reference output exists
    assert h["natural_bugs_eval_also_catches_strict"] is True

    # the structural blind spot: corrupted state, identical forward pass
    assert sorted(h["forward_blind_corrupting_cases"]) == [
        "progress_reset", "stale_optimizer_moment", "step_counter_reset"]
    assert h["forward_blind_all_caught_by_certificate"] is True
    for name in h["forward_blind_corrupting_cases"]:
        assert cases[name]["output_max_rel_dev"] == 0.0
        assert not cases[name]["certificate_committed"]

    # near-tolerance damage: below any usable eval threshold, still aborted
    assert cases["precision_cast"]["eval_catches_strict"] is False
    assert not cases["precision_cast"]["certificate_committed"]
    assert cases["value_corruption"]["eval_catches_realistic"] is False
    assert not cases["value_corruption"]["certificate_committed"]

    # clean arms stay clean on both oracles
    for name in ("zero_to_fp32_fixed", "swiglu_reshard_fixed"):
        assert cases[name]["certificate_committed"]
        assert cases[name]["output_max_rel_dev"] == 0.0
