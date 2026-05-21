"""Thin client for ElectricityMaps and WattTime carbon-intensity APIs.

Only used in real-deployment mode (Phase 3+); offline experiments should plug in
``CarbonTrace`` replay instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import requests  # optional dep
except ImportError:  # pragma: no cover
    requests = None


@dataclass
class ElectricityMapsClient:
    api_key: str
    zone: str = "VN"
    base_url: str = "https://api.electricitymap.org/v3"

    @classmethod
    def from_env(cls, zone: str | None = None) -> ElectricityMapsClient:
        key = os.environ.get("ELECTRICITYMAPS_API_KEY")
        if not key:
            raise RuntimeError("Set ELECTRICITYMAPS_API_KEY in the environment.")
        return cls(api_key=key, zone=zone or os.environ.get("ELECTRICITYMAPS_ZONE", "VN"))

    def carbon_intensity_now(self) -> float:
        if requests is None:
            raise RuntimeError("Install `requests` (pip install hise[energy-api])")
        url = f"{self.base_url}/carbon-intensity/latest"
        resp = requests.get(url, params={"zone": self.zone},
                            headers={"auth-token": self.api_key}, timeout=10.0)
        resp.raise_for_status()
        return float(resp.json()["carbonIntensity"])

    def carbon_intensity_forecast(self, horizon_hours: int = 24) -> list[tuple[float, float]]:
        """Returns ``[(seconds_from_now, gCO2/kWh), ...]``."""
        if requests is None:
            raise RuntimeError("Install `requests` (pip install hise[energy-api])")
        url = f"{self.base_url}/carbon-intensity/forecast"
        resp = requests.get(url, params={"zone": self.zone},
                            headers={"auth-token": self.api_key}, timeout=10.0)
        resp.raise_for_status()
        out: list[tuple[float, float]] = []
        for i, point in enumerate(resp.json().get("forecast", [])):
            out.append((i * 3600.0, float(point["carbonIntensity"])))
            if i * 3600.0 >= horizon_hours * 3600.0:
                break
        return out
