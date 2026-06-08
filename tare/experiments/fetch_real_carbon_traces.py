"""Fetch real grid carbon-intensity traces from the Energy-Charts public API.

Energy-Charts (Fraunhofer ISE, https://api.energy-charts.info) exposes
hourly-or-sub-hourly CO2eq for European zones via a free REST endpoint
backed by ENTSO-E data. This script downloads a configurable window per
zone, resamples to the harness's hourly cadence, and writes
ElectricityMaps-compatible CSVs that ``load_electricitymaps_csv`` can
read directly. The schema written here matches
``tests/fixtures/electricitymaps_de_7day_sample.csv`` so the downstream
consumer needs no changes.

Coverage today:
  - **DE** (Germany) — native 15-min resolution
  - **NO** (Norway) — native 30-min resolution
  - **ZA** (South Africa) — not on Energy-Charts; user must supply an
    ElectricityMaps CSV manually for ZA if it is needed.

Resampling: mean over each ``--sample-minutes`` window. Default 60 min
matches the H5-C harness window cadence.

Usage::

    python -m experiments.fetch_real_carbon_traces \\
        --zones DE NO --start 2024-07-01 --end 2024-07-15 \\
        --out data_cache/real_traces
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ENERGY_CHARTS_URL = "https://api.energy-charts.info/co2eq"
# GB left the ENTSO-E transparency platform after Brexit, so Energy-Charts no
# longer serves its co2eq. GB has its own free, no-auth official API instead.
# Note: this endpoint reports *operational* carbon intensity (gCO2/kWh), a
# slightly different basis from Energy-Charts' LCA co2eq; both are gCO2/kWh and
# fine for the relative carbon-shift analysis, but the basis is not identical.
UK_CARBON_INTENSITY_URL = "https://api.carbonintensity.org.uk/intensity"

# Energy-Charts uses lower-case ISO 3166-1 alpha-2 country codes; the
# value-add here is a stable mapping back to the ElectricityMaps zone
# code so the saved file plugs straight into the H5-C harness.
ENERGY_CHARTS_ZONES = {
    "DE": ("de", "Germany"),
    "NO": ("no", "Norway"),
    "FR": ("fr", "France"),
    "GB": ("gb", "Great Britain"),
    "PL": ("pl", "Poland"),
}


def fetch_energy_charts(zone_em: str, start: datetime, end: datetime) -> tuple[list[datetime], list[float]]:
    """Return (timestamps, co2eq_g_per_kwh) for ``zone_em`` over ``[start, end)``."""
    if zone_em not in ENERGY_CHARTS_ZONES:
        raise ValueError(
            f"zone {zone_em!r} not on Energy-Charts. Supported: "
            f"{sorted(ENERGY_CHARTS_ZONES)}. For other zones (e.g., ZA), "
            "supply an ElectricityMaps CSV manually."
        )
    code, _ = ENERGY_CHARTS_ZONES[zone_em]
    params = {
        "country": code,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
    }
    url = f"{ENERGY_CHARTS_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    import json
    data = json.loads(payload)
    ts = [datetime.fromtimestamp(s, tz=UTC) for s in data["unix_seconds"]]
    return ts, data["co2eq"]


def fetch_uk_carbon_intensity(zone_em: str, start: datetime, end: datetime) -> tuple[list[datetime], list[float]]:
    """Return (timestamps, gCO2/kWh) for GB from the UK Carbon Intensity API.

    The ``/intensity/{from}/{to}`` endpoint caps the range per request, so the
    window is fetched in 7-day chunks and concatenated. Uses the measured
    ``actual`` intensity, falling back to ``forecast`` for any half-hour the
    settlement run had not yet finalised.
    """
    import json
    from datetime import timedelta

    if zone_em != "GB":
        raise ValueError(f"UK Carbon Intensity API serves GB only, not {zone_em!r}")
    ts_all: list[datetime] = []
    val_all: list[float] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=7), end)
        frm = cur.strftime("%Y-%m-%dT%H:%MZ")
        to = nxt.strftime("%Y-%m-%dT%H:%MZ")
        url = f"{UK_CARBON_INTENSITY_URL}/{frm}/{to}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read())
        for row in data.get("data", []):
            inten = row.get("intensity") or {}
            v = inten.get("actual")
            if v is None:
                v = inten.get("forecast")
            raw_t = row.get("from")
            if v is None or raw_t is None:
                continue
            t = datetime.fromisoformat(raw_t.replace("Z", "+00:00")).astimezone(UTC)
            ts_all.append(t)
            val_all.append(float(v))
        cur = nxt
    return ts_all, val_all


def fetch_zone(zone_em: str, start: datetime, end: datetime) -> tuple[list[datetime], list[float]]:
    """Dispatch to the right free source for a zone (GB → UK API; else Energy-Charts)."""
    if zone_em == "GB":
        return fetch_uk_carbon_intensity(zone_em, start, end)
    return fetch_energy_charts(zone_em, start, end)


def resample_to_hourly(
    timestamps: list[datetime],
    intensities: list[float],
    sample_minutes: int = 60,
) -> tuple[list[datetime], list[float]]:
    """Aggregate sub-hourly samples into ``sample_minutes`` buckets by mean."""
    if not timestamps:
        return ([], [])
    bucket_seconds = sample_minutes * 60
    buckets: dict[int, list[float]] = {}
    bucket_keys: dict[int, datetime] = {}
    epoch0 = int(timestamps[0].timestamp()) // bucket_seconds * bucket_seconds
    for ts, val in zip(timestamps, intensities, strict=True):
        key = (int(ts.timestamp()) - epoch0) // bucket_seconds
        buckets.setdefault(key, []).append(val)
        bucket_keys.setdefault(key, datetime.fromtimestamp(epoch0 + key * bucket_seconds, tz=UTC))
    out_ts: list[datetime] = []
    out_int: list[float] = []
    for key in sorted(buckets):
        out_ts.append(bucket_keys[key])
        out_int.append(statistics.mean(buckets[key]))
    return out_ts, out_int


def write_electricitymaps_csv(
    path: Path,
    zone_em: str,
    timestamps: list[datetime],
    intensities: list[float],
) -> None:
    """Write a CSV in the schema ``load_electricitymaps_csv`` expects."""
    _, country = ENERGY_CHARTS_ZONES.get(zone_em, ("", zone_em))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Datetime (UTC)", "Country", "Zone Name", "Zone Id",
            "Carbon Intensity gCO₂eq/kWh (LCA)",
        ])
        for ts, val in zip(timestamps, intensities, strict=True):
            writer.writerow([
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                country, country, zone_em, f"{val:.2f}",
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones", nargs="+", default=["DE", "NO"])
    parser.add_argument("--start", default="2024-07-01", help="Inclusive start date.")
    parser.add_argument("--end", default="2024-07-15", help="Exclusive end date.")
    parser.add_argument("--sample-minutes", type=int, default=60)
    parser.add_argument("--out", default="data_cache/real_traces")
    parser.add_argument("--rate-limit-sleep-s", type=float, default=10.0,
                        help="Pause between zone fetches to respect Energy-Charts rate limit.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    days = (end - start).days
    print(f"Fetching {days}-day window {args.start} → {args.end} for zones {args.zones}")

    for i, zone in enumerate(args.zones):
        if i > 0:
            time.sleep(args.rate_limit_sleep_s)
        print(f"  {zone}: …", end="", flush=True)
        try:
            ts, vals = fetch_zone(zone, start, end)
        except urllib.error.HTTPError as e:
            print(f" FAIL ({e.code}); skipping")
            continue
        ts_h, vals_h = resample_to_hourly(ts, vals, args.sample_minutes)
        target = out_dir / f"{zone.lower()}_{args.start}_{args.end}_hourly.csv"
        write_electricitymaps_csv(target, zone, ts_h, vals_h)
        # Span check: at hourly cadence, expected = days × 24 samples ± 1.
        expected = days * (60 // args.sample_minutes) * 24
        print(f" wrote {len(ts_h)} rows (expected ~{expected}) → {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
