"""Tests for the scale-N warm-keep kill-gate (pure analysis, no GPU).

These lock the construct invariants the gate relies on: every policy burns the
same compute joules (only the wall-clock hours differ), the reshard brackets are
ordered, and -- the decisive physical fact -- carbon-aware scale-N cannot beat
full pausing on carbon because pause evacuates dirty hours entirely while
scale-N keeps doing dirty-hour work."""
from __future__ import annotations

from experiments.exp_scalen_warmkeep_breakeven import (
    reload_tax,
    replay,
    reshard_tax,
)

_RELOAD_KW = dict(cold_start_s=4.7, resume_power_per_gpu_w=100.0,
                  write_bw_gbps=3.9, reload_bw_gbps=7.7, warmup_s=0.761)
# a clean diurnal shape: clean nights, dirty days (threshold sits mid-range)
_DIURNAL = [200.0, 200.0, 900.0, 900.0] * 6   # 24 h, 50% dirty above thr=500
_THR = 500.0


def _common(state_gb: float = 20.0, n: int = 8, m: int = 4):
    reload_t = reload_tax(state_gb, n=n, **_RELOAD_KW)
    reshard_t = reshard_tax(state_gb, n=n, m=m, bracket="lower", collective_bw_gbps=200.0,
                            dynatrain_const_s=4.36, resume_power_per_gpu_w=100.0, reload_kwargs=_RELOAD_KW)
    return dict(start_hour=0, h_job_fngh=12.0, threshold=_THR, n=n, m=m,
                e_fngh_kwh=2.4, reload_t=reload_t, reshard_t=reshard_t)


def test_compute_joules_invariant_across_policies():
    """The construct guard: every policy does the same work => same compute kWh."""
    common = _common()
    outs = {p: replay(_DIURNAL, p, **common)
            for p in ("always_on", "reduced_n", "pause", "scale_n")}
    compute = [o["compute_kwh"] for o in outs.values()]
    assert max(compute) - min(compute) < 1e-9
    # and it equals the full job work x per-FNGH energy
    assert abs(compute[0] - 12.0 * 2.4) < 1e-9


def test_reshard_brackets_ordered():
    """lower (warm, moved fraction) <= dynatrain (fast const) <= upper (= full reload)."""
    kw = dict(n=8, m=4, collective_bw_gbps=200.0, dynatrain_const_s=4.36,
              resume_power_per_gpu_w=100.0, reload_kwargs=_RELOAD_KW)
    big = 160.0
    lo = reshard_tax(big, bracket="lower", **kw).time_s
    dyna = reshard_tax(big, bracket="dynatrain", **kw).time_s
    up = reshard_tax(big, bracket="upper", **kw).time_s
    full_reload = reload_tax(big, n=8, **_RELOAD_KW).time_s
    assert lo <= up and dyna <= up
    assert abs(up - full_reload) < 1e-9          # upper bracket == pause's reload
    assert lo < full_reload                       # warm reshard strictly cheaper than full reload


def test_reload_tax_grows_with_state():
    small = reload_tax(1.0, n=8, **_RELOAD_KW).time_s
    big = reload_tax(160.0, n=8, **_RELOAD_KW).time_s
    assert big > small > 4.7   # both exceed cold start; larger state costs more


def test_scale_n_cannot_beat_pause_on_carbon():
    """The decisive negative: pause evacuates dirty hours entirely, scale-N keeps
    doing reduced-rate dirty-hour work, so scale-N's carbon is >= pause's even with
    the optimistic (cheapest) reshard tax and even at large state."""
    for state_gb in (1.0, 80.0, 160.0):
        common = _common(state_gb=state_gb)
        pa = replay(_DIURNAL, "pause", **common)
        sc = replay(_DIURNAL, "scale_n", **common)
        assert sc["carbon_g"] > pa["carbon_g"]      # scale-N loses to pause on carbon
        assert sc["makespan_h"] < pa["makespan_h"]  # but finishes sooner (a different Pareto point)


def test_scale_n_beats_carbon_blind_reduced_n_on_carbon():
    """Awareness component is real and positive: concentrating full-count work in
    clean hours beats running a constant reduced count (the CarbonScaler effect)."""
    common = _common()
    rn = replay(_DIURNAL, "reduced_n", **common)
    sc = replay(_DIURNAL, "scale_n", **common)
    assert sc["carbon_g"] < rn["carbon_g"]


def test_tax_energy_billed_but_not_counted_as_compute():
    """Transition energy is real (raises total energy + carbon) but is not compute."""
    common = _common(state_gb=80.0)
    sc = replay(_DIURNAL, "scale_n", **common)
    assert sc["tax_kwh"] > 0.0
    assert sc["switches"] > 0
    assert abs(sc["energy_kwh"] - (sc["compute_kwh"] + sc["tax_kwh"])) < 1e-12
    # always_on never switches, pays no tax
    ao = replay(_DIURNAL, "always_on", **common)
    assert ao["tax_kwh"] == 0.0 and ao["switches"] == 0


def test_zero_tax_when_no_dirty_hours():
    """An all-clean trace never crosses a boundary: no pause, no reshard, no tax."""
    clean = [100.0] * 24
    common = _common()
    for p in ("pause", "scale_n"):
        o = replay(clean, p, **common, )
        assert o["switches"] == 0 and o["tax_kwh"] == 0.0
        assert o["makespan_h"] == 12.0   # runs straight through at full count
