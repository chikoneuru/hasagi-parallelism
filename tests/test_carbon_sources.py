"""Tests for multi-source carbon proxy clients + aggregator.

Uses dependency injection (fake http_client) so the suite never hits real EM
or WattTime endpoints during CI.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from hasagi.energy.carbon_sources import (
    AggregateCarbonReading,
    CarbonReading,
    ElectricityMapsClient,
    IEAStaticSource,
    MultiSourceCarbonAggregator,
    WattTimeClient,
)

# --- Test doubles ---

@dataclass
class _FakeResponse:
    payload: dict
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self.payload


class _FakeHttpClient:
    """Captures call args + returns canned response. Stateless across requests."""

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, headers: dict | None = None) -> _FakeResponse:
        self.calls.append((url, headers or {}))
        return self.response

    def close(self) -> None:  # never called when injected (cleanup path skipped)
        pass


# --- IEAStaticSource ---

def test_iea_static_returns_known_region() -> None:
    src = IEAStaticSource(region="DE")
    r = src.read()
    assert r.source == "iea_static"
    assert r.region == "DE"
    assert r.intensity_g_per_kwh == 380.0
    assert r.uncertainty_g_per_kwh == 380.0 * 0.25
    assert r.methodology == "average"


def test_iea_static_rejects_unknown_region() -> None:
    with pytest.raises(ValueError, match="not in IEA static table"):
        IEAStaticSource(region="ZZ")


def test_iea_static_custom_table_supported() -> None:
    custom = {"X1": 100.0, "X2": 500.0}
    src = IEAStaticSource(region="X1", table=custom, uncertainty_fraction=0.1)
    r = src.read()
    assert r.intensity_g_per_kwh == 100.0
    assert r.uncertainty_g_per_kwh == 10.0


def test_iea_static_extreme_regions_cover_h5c_span() -> None:
    """Carbon-claim spans 40× intensity (IS to US-WV); ensure both endpoints exist."""
    low = IEAStaticSource(region="IS").read().intensity_g_per_kwh
    high = IEAStaticSource(region="US-WV").read().intensity_g_per_kwh
    assert high / low > 40   # 850 / 12 ≈ 70


# --- ElectricityMapsClient ---

def test_em_client_parses_intensity() -> None:
    http = _FakeHttpClient(_FakeResponse({"carbonIntensity": 250.0, "datetime": "2026-05-22T12:00Z"}))
    client = ElectricityMapsClient(api_key="key", region="DE", http_client=http)
    r = client.read()
    assert r.source == "electricitymaps"
    assert r.region == "DE"
    assert r.intensity_g_per_kwh == 250.0
    assert r.uncertainty_g_per_kwh == 250.0 * 0.125
    assert r.methodology == "average"


def test_em_client_sends_auth_token_header() -> None:
    http = _FakeHttpClient(_FakeResponse({"carbonIntensity": 100.0}))
    client = ElectricityMapsClient(api_key="my-key", region="VN", http_client=http)
    client.read()
    assert len(http.calls) == 1
    url, headers = http.calls[0]
    assert "zone=VN" in url
    assert headers["auth-token"] == "my-key"


def test_em_client_propagates_http_errors() -> None:
    http = _FakeHttpClient(_FakeResponse({}, status_code=500))
    client = ElectricityMapsClient(api_key="key", region="DE", http_client=http)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        client.read()


# --- WattTimeClient ---

def test_watttime_client_converts_lbs_per_mwh_to_g_per_kwh() -> None:
    """MOER 1000 lbs/MWh → 453.592 g/kWh."""
    http = _FakeHttpClient(_FakeResponse({"data": [{"value": 1000.0}]}))
    client = WattTimeClient(token="tok", region="CAISO_NORTH", http_client=http)
    r = client.read()
    assert r.source == "watttime"
    assert r.methodology == "marginal"
    assert abs(r.intensity_g_per_kwh - 453.592) < 1e-3


def test_watttime_uses_bearer_auth() -> None:
    http = _FakeHttpClient(_FakeResponse({"data": [{"value": 800.0}]}))
    client = WattTimeClient(token="tok123", region="PJM", http_client=http)
    client.read()
    _, headers = http.calls[0]
    assert headers["Authorization"] == "Bearer tok123"


# --- MultiSourceCarbonAggregator ---

class _StubSource:
    def __init__(self, intensity: float, *, source: str = "stub",
                 region: str = "DE", uncertainty: float | None = None,
                 methodology: str = "average") -> None:
        self.intensity = intensity
        self.source = source
        self.region = region
        self.uncertainty = uncertainty if uncertainty is not None else intensity * 0.1
        self.methodology = methodology

    def read(self) -> CarbonReading:
        import time
        return CarbonReading(
            source=self.source, region=self.region,
            intensity_g_per_kwh=self.intensity,
            uncertainty_g_per_kwh=self.uncertainty,
            timestamp_s=time.time(),
            methodology=self.methodology,
        )


def test_aggregator_combines_sources_into_mean() -> None:
    agg = MultiSourceCarbonAggregator(
        sources=[_StubSource(100.0), _StubSource(200.0), _StubSource(300.0)],
    )
    r = agg.read()
    assert r.intensity_g_per_kwh == 200.0
    assert r.min_g_per_kwh == 100.0
    assert r.max_g_per_kwh == 300.0
    assert len(r.sources_used) == 3


def test_aggregator_flags_disagreement_above_threshold() -> None:
    """3 sources spanning 100..500 → disagreement (500-100)/300 ≈ 1.33 → flagged."""
    agg = MultiSourceCarbonAggregator(
        sources=[_StubSource(100.0), _StubSource(300.0), _StubSource(500.0)],
        flag_threshold=0.20,
    )
    r = agg.read()
    assert r.disagreement_flagged
    assert r.disagreement_fraction > 0.20


def test_aggregator_does_not_flag_close_sources() -> None:
    """3 sources within 10% spread → not flagged."""
    agg = MultiSourceCarbonAggregator(
        sources=[_StubSource(100.0), _StubSource(105.0), _StubSource(110.0)],
        flag_threshold=0.20,
    )
    r = agg.read()
    assert not r.disagreement_flagged


def test_aggregator_methodologies_reports_marginal_vs_average() -> None:
    """When sources mix average + marginal, both labels appear in methodologies."""
    agg = MultiSourceCarbonAggregator(
        sources=[
            _StubSource(100.0, source="em", methodology="average"),
            _StubSource(150.0, source="wt", methodology="marginal"),
            _StubSource(120.0, source="iea", methodology="average"),
        ],
    )
    r = agg.read()
    assert set(r.methodologies) == {"average", "marginal"}


class _FailingSource:
    def read(self) -> CarbonReading:
        raise RuntimeError("simulated API failure")


def test_aggregator_drops_failed_sources_gracefully() -> None:
    """One source fails out of 3 → aggregate still returns from the remaining 2."""
    agg = MultiSourceCarbonAggregator(
        sources=[_StubSource(100.0), _FailingSource(), _StubSource(200.0)],
    )
    r = agg.read()
    assert r.intensity_g_per_kwh == 150.0
    assert len(r.sources_used) == 2


def test_aggregator_raises_when_all_sources_fail() -> None:
    agg = MultiSourceCarbonAggregator(sources=[_FailingSource(), _FailingSource()])
    with pytest.raises(RuntimeError, match="All carbon sources failed"):
        agg.read()


def test_aggregator_uncertainty_is_root_sum_square() -> None:
    """RSS combination: σ = sqrt(σ1² + σ2² + σ3²)."""
    agg = MultiSourceCarbonAggregator(
        sources=[
            _StubSource(100.0, uncertainty=10.0),
            _StubSource(110.0, uncertainty=15.0),
        ],
    )
    r = agg.read()
    expected = (10.0**2 + 15.0**2) ** 0.5
    assert abs(r.uncertainty_g_per_kwh - expected) < 1e-9


def test_aggregator_end_to_end_with_real_clients() -> None:
    """Full path: 3 source types (EM/WattTime/IEA) → aggregate. Uses fake HTTP."""
    em_http = _FakeHttpClient(_FakeResponse({"carbonIntensity": 200.0}))
    wt_http = _FakeHttpClient(_FakeResponse({"data": [{"value": 500.0}]}))  # → 226.8 g/kWh
    sources = [
        ElectricityMapsClient(api_key="k1", region="DE", http_client=em_http),
        WattTimeClient(token="t1", region="DE", http_client=wt_http),
        IEAStaticSource(region="DE"),    # = 380
    ]
    agg = MultiSourceCarbonAggregator(sources=sources)
    r = agg.read()
    assert isinstance(r, AggregateCarbonReading)
    assert len(r.sources_used) == 3
    # Range covers EM ~200 → IEA 380.
    assert r.min_g_per_kwh < 230
    assert r.max_g_per_kwh > 370
