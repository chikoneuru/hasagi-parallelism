"""Multi-source carbon intensity proxy with uncertainty propagation.

Carbon proxy is reported from **three independent sources in parallel**;
disagreement >20% between sources is flagged explicitly rather than cherry-picked:

* **ElectricityMaps** — 1h granularity, ±10-15% accuracy, average emissions.
* **WattTime** — 5-min granularity, ±20% accuracy, marginal emissions.
* **IEA static** — yearly average per region, no API needed.

The aggregator returns a ``CarbonReading`` with mean + bounds; downstream
(carbon claim evaluation, MPC weighting) uses the band, not a single number.

API clients are dependency-injectable so tests run without real HTTP:
each client accepts a ``http_client`` parameter that defaults to ``httpx``
but tests pass a stub returning canned responses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CarbonReading:
    """One carbon-intensity sample from a named source.

    Attributes:
        source: human-readable source name (``"electricitymaps"``, ``"watttime"``,
            ``"iea_static"``).
        region: ISO region code (``"VN"``, ``"DE"``, ``"US-CA"``, ...).
        intensity_g_per_kwh: gCO2eq per kWh of electricity at this region.
        uncertainty_g_per_kwh: 1-σ uncertainty per the source's published bound
            (EM ±10-15%, WattTime ±20%, IEA static unknown → use 0.25 * value).
        timestamp_s: monotonic seconds, ``time.time()``-style.
        methodology: ``"average"`` (EM, IEA) vs ``"marginal"`` (WattTime).
    """

    source: str
    region: str
    intensity_g_per_kwh: float
    uncertainty_g_per_kwh: float
    timestamp_s: float
    methodology: str = "average"


class CarbonSource(Protocol):
    """All carbon sources expose a single ``read()`` method."""

    def read(self) -> CarbonReading: ...


# ---------------------------------------------------------------------------
# IEA static — hardcoded yearly averages, no API
# ---------------------------------------------------------------------------

# 2022 grid emission factors from IEA Energy Statistics, in gCO2/kWh.
# Source: IEA "CO2 Emissions from Fuel Combustion" 2022 edition. Values are
# averages; uncertainty assumed 25% (yearly aggregation hides intraday swings).
_IEA_STATIC_2022: dict[str, float] = {
    "IS": 12.0,    # Iceland — geothermal/hydro
    "PL": 720.0,   # Poland — coal heavy
    "DE": 380.0,   # Germany — mixed
    "VN": 470.0,   # Vietnam — coal + hydro
    "US-CA": 200.0,  # California
    "US-WV": 850.0,  # West Virginia — coal heavy
    "FR": 80.0,    # France — nuclear
    "BR": 90.0,    # Brazil — hydro dominant
    "JP": 480.0,   # Japan
    "CN": 580.0,   # China
    "GB": 290.0,   # UK
}


@dataclass
class IEAStaticSource:
    """Constant-intensity source from IEA 2022 yearly averages.

    Used as the third source for carbon-claim cross-validation and as a no-network
    fallback when the live APIs are unavailable. Always returns the same
    intensity per ``read()``; uncertainty is fixed at 25% per IEA aggregation
    methodology (yearly average hides intraday swings).
    """

    region: str
    table: dict[str, float] | None = None
    uncertainty_fraction: float = 0.25

    def __post_init__(self) -> None:
        table = self.table if self.table is not None else _IEA_STATIC_2022
        if self.region not in table:
            raise ValueError(
                f"Region {self.region!r} not in IEA static table. "
                f"Known regions: {sorted(table.keys())}"
            )
        self._intensity = table[self.region]

    def read(self) -> CarbonReading:
        import time
        return CarbonReading(
            source="iea_static",
            region=self.region,
            intensity_g_per_kwh=self._intensity,
            uncertainty_g_per_kwh=self._intensity * self.uncertainty_fraction,
            timestamp_s=time.time(),
            methodology="average",
        )


# ---------------------------------------------------------------------------
# ElectricityMaps live API
# ---------------------------------------------------------------------------

@dataclass
class ElectricityMapsClient:
    """Live ElectricityMaps API client. Requires an API key (free academic tier).

    Endpoint: ``https://api.electricitymap.org/v3/carbon-intensity/latest?zone={region}``
    Returns JSON ``{"carbonIntensity": ..., "datetime": ..., ...}`` in gCO2/kWh.
    Accuracy ±10-15% per the EM methodology paper.

    Args:
        api_key: EM auth token.
        region: ISO region code matching EM zones (e.g., ``"DE"``, ``"US-CAL-CISO"``).
        http_client: optional HTTP client with ``.get(url, headers)`` method
            returning an object with ``.json()`` + ``.raise_for_status()``.
            Defaults to ``httpx.Client``. Tests pass a stub.
        timeout_s: HTTP request timeout (default 5s).
    """

    api_key: str
    region: str
    http_client: Any = None
    timeout_s: float = 5.0
    uncertainty_fraction: float = 0.125   # midpoint of EM's ±10-15% band

    def _client(self) -> Any:
        if self.http_client is not None:
            return self.http_client
        import httpx
        return httpx.Client(timeout=self.timeout_s)

    def read(self) -> CarbonReading:
        import time
        url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={self.region}"
        headers = {"auth-token": self.api_key}
        client = self._client()
        try:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        finally:
            if self.http_client is None:
                client.close()
        intensity = float(data["carbonIntensity"])
        return CarbonReading(
            source="electricitymaps",
            region=self.region,
            intensity_g_per_kwh=intensity,
            uncertainty_g_per_kwh=intensity * self.uncertainty_fraction,
            timestamp_s=time.time(),
            methodology="average",
        )


# ---------------------------------------------------------------------------
# WattTime live API
# ---------------------------------------------------------------------------

@dataclass
class WattTimeClient:
    """Live WattTime API client. Requires registration (free academic tier).

    Endpoint: ``https://api.watttime.org/v3/signal-index?region={region}``
    Returns MOER (Marginal Operating Emissions Rate) in lbs/MWh; we convert
    to gCO2/kWh via ``lbs/MWh × 453.592 / 1000 = g/kWh``.

    Marginal methodology differs from EM's average — for HASAGI's spatial-shift
    framing, marginal is the theoretically-correct metric (what emissions
    would have been avoided if this kWh wasn't used here). HASAGI reports both
    side-by-side and flags disagreement >20% explicitly.

    Args:
        token: WattTime auth token (Bearer).
        region: WattTime region code (e.g., ``"CAISO_NORTH"``, ``"PJM"``).
        http_client: optional injected HTTP client for testing.
        timeout_s: request timeout (default 5s).
    """

    token: str
    region: str
    http_client: Any = None
    timeout_s: float = 5.0
    uncertainty_fraction: float = 0.20   # WattTime ±20% per their methodology debate

    def _client(self) -> Any:
        if self.http_client is not None:
            return self.http_client
        import httpx
        return httpx.Client(timeout=self.timeout_s)

    def read(self) -> CarbonReading:
        import time
        url = f"https://api.watttime.org/v3/signal-index?region={self.region}"
        headers = {"Authorization": f"Bearer {self.token}"}
        client = self._client()
        try:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        finally:
            if self.http_client is None:
                client.close()
        # MOER lbs/MWh → gCO2/kWh
        moer_lbs_per_mwh = float(data["data"][0]["value"])
        intensity = moer_lbs_per_mwh * 453.592 / 1000.0
        return CarbonReading(
            source="watttime",
            region=self.region,
            intensity_g_per_kwh=intensity,
            uncertainty_g_per_kwh=intensity * self.uncertainty_fraction,
            timestamp_s=time.time(),
            methodology="marginal",
        )


# ---------------------------------------------------------------------------
# MultiSourceAggregator — combine + flag disagreement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregateCarbonReading:
    """Per-tick carbon proxy with explicit uncertainty bounds.

    ``intensity_g_per_kwh`` is the mean of all source readings; ``min_g_per_kwh``
    / ``max_g_per_kwh`` bound the band. ``disagreement_fraction`` = ``(max-min)/mean``
    is HASAGI's flag for "report range, do not cherry-pick" — values above
    ``flag_threshold`` (default 0.20) trigger a logged warning.

    Methodologies present in the inputs are listed in ``methodologies`` so
    downstream knows whether the average mixed marginal + average sources.
    """

    intensity_g_per_kwh: float
    min_g_per_kwh: float
    max_g_per_kwh: float
    uncertainty_g_per_kwh: float
    disagreement_fraction: float
    sources_used: tuple[str, ...]
    methodologies: tuple[str, ...]
    timestamp_s: float
    disagreement_flagged: bool = False


@dataclass
class MultiSourceCarbonAggregator:
    """Combine ``CarbonSource`` readings with disagreement flagging.

    Per-tick flow::

        agg = MultiSourceCarbonAggregator(sources=[em, wt, iea])
        reading = agg.read()
        if reading.disagreement_flagged:
            # log + write to paper sensitivity appendix
            ...
        ctrl.use_intensity(reading.intensity_g_per_kwh)

    Failed sources (network errors, missing region) are dropped from the
    aggregate; if ALL sources fail, raises RuntimeError. With 3 sources and
    1 failure the aggregate still passes — graceful degradation.
    """

    sources: list[CarbonSource]
    flag_threshold: float = 0.20

    def read(self) -> AggregateCarbonReading:
        readings: list[CarbonReading] = []
        for src in self.sources:
            try:
                readings.append(src.read())
            except Exception:
                logger.warning("Carbon source %s failed; dropping",
                                type(src).__name__, exc_info=True)
        if not readings:
            raise RuntimeError("All carbon sources failed")

        values = [r.intensity_g_per_kwh for r in readings]
        mean = sum(values) / len(values)
        lo = min(values)
        hi = max(values)
        disagreement = (hi - lo) / mean if mean > 0 else 0.0
        # Combined uncertainty: pessimistic — root-sum-square of per-source
        # bounds. Caller can derive the full band via [mean - U, mean + U].
        uncertainty = (sum(r.uncertainty_g_per_kwh ** 2 for r in readings)) ** 0.5
        flagged = disagreement > self.flag_threshold
        if flagged:
            logger.warning(
                "Carbon sources disagree by %.1f%% (threshold %.0f%%); "
                "mean=%.1f range=[%.1f, %.1f]",
                disagreement * 100, self.flag_threshold * 100, mean, lo, hi,
            )

        return AggregateCarbonReading(
            intensity_g_per_kwh=mean,
            min_g_per_kwh=lo,
            max_g_per_kwh=hi,
            uncertainty_g_per_kwh=uncertainty,
            disagreement_fraction=disagreement,
            sources_used=tuple(r.source for r in readings),
            methodologies=tuple(sorted({r.methodology for r in readings})),
            timestamp_s=max(r.timestamp_s for r in readings),
            disagreement_flagged=flagged,
        )
