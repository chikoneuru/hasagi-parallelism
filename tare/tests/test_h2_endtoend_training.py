"""Unit tests for the H2 end-to-end training harness helpers.

The actual training loop requires CUDA + torchvision; those paths are not
exercised here. We pin the no-CUDA helpers (summariser, JSON I/O,
``set_power_cap`` best-effort) so the harness's reporting layer can be
refactored safely without re-running the full multi-seed training sweep.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.exp_h2_endtoend_training import (
    CONDITIONS,
    EpochSample,
    RunResult,
    _summarise,
    _write_result,
    set_power_cap,
)


def _make_run(seed: int, condition: str, final_top1: float, energy_j: float,
              wall_s: float = 600.0) -> RunResult:
    return RunResult(
        seed=seed,
        condition=condition,
        epochs=30,
        batch_size=128,
        learning_rate=0.1,
        final_top1=final_top1,
        total_wall_seconds=wall_s,
        total_energy_joules=energy_j,
        per_epoch=[EpochSample(0, 1.0, final_top1, wall_s, energy_j, energy_j / wall_s)],
    )


def test_conditions_constant_is_canonical() -> None:
    """The harness enumerates exactly these four conditions; reporting iterates over the tuple."""
    assert CONDITIONS == ("static", "dvfs", "preempt", "combined")


def test_write_result_creates_directory(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir"
    r = _make_run(0, "static", 92.0, 1e6)
    path = _write_result(out, r)
    assert path.exists()
    assert out.is_dir()


def test_write_result_filename_format(tmp_path: Path) -> None:
    r = _make_run(7, "dvfs", 91.5, 1e6)
    path = _write_result(tmp_path, r)
    assert path.name == "seed7_dvfs.json"


def test_write_result_roundtrips(tmp_path: Path) -> None:
    r = _make_run(3, "preempt", 90.8, 1.5e6, wall_s=720.0)
    path = _write_result(tmp_path, r)
    data = json.loads(path.read_text())
    assert data["seed"] == 3
    assert data["condition"] == "preempt"
    assert data["final_top1"] == 90.8
    assert data["total_energy_joules"] == 1.5e6


def test_summarise_aggregates_by_condition(tmp_path: Path) -> None:
    runs = [
        _make_run(0, "static", 92.0, 1.0e6),
        _make_run(1, "static", 91.5, 1.05e6),
        _make_run(2, "static", 92.5, 0.95e6),
        _make_run(0, "dvfs", 91.8, 0.85e6),
        _make_run(1, "dvfs", 91.6, 0.88e6),
    ]
    for r in runs:
        _write_result(tmp_path, r)
    summary = _summarise(tmp_path)
    assert summary["static"]["n"] == 3
    assert abs(summary["static"]["mean_top1"] - 92.0) < 1e-9
    assert summary["static"]["min_top1"] == 91.5
    assert summary["static"]["max_top1"] == 92.5
    assert summary["dvfs"]["n"] == 2
    assert abs(summary["dvfs"]["mean_top1"] - 91.7) < 1e-9


def test_summarise_converts_joules_to_kwh(tmp_path: Path) -> None:
    _write_result(tmp_path, _make_run(0, "static", 92.0, 3_600_000.0))
    summary = _summarise(tmp_path)
    # 3.6e6 J = 1 kWh exactly.
    assert abs(summary["static"]["mean_energy_kwh"] - 1.0) < 1e-9


def test_summarise_converts_seconds_to_minutes(tmp_path: Path) -> None:
    _write_result(tmp_path, _make_run(0, "static", 92.0, 1.0e6, wall_s=300.0))
    summary = _summarise(tmp_path)
    assert abs(summary["static"]["mean_wall_minutes"] - 5.0) < 1e-9


def test_summarise_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    """No matching artifacts → empty summary, no crash."""
    summary = _summarise(tmp_path / "does_not_exist")
    assert summary == {}


def test_set_power_cap_is_best_effort(tmp_path: Path, monkeypatch) -> None:
    """When the sudo-nvidia-smi binary is unavailable, set_power_cap must not raise."""
    import subprocess

    def fake_run(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi not found in PATH")

    monkeypatch.setattr(subprocess, "run", fake_run)
    set_power_cap(200)   # no exception
