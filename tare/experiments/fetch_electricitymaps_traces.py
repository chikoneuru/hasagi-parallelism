"""Fetch all 16 zones from ONE source — the ElectricityMaps API — for a unified set.

Energy-Charts / the UK API only cover European zones; ElectricityMaps is the
single platform with global hourly coverage of the full 16-zone set, under one
LCA methodology. This pulls the historical hourly carbon intensity per zone via
the v3 ``carbon-intensity/past-range`` endpoint and writes the same EM-compatible
CSV schema the harness already reads (``load_zone_traces`` picks them up by the
``<zone>_*.csv`` name with zero code changes).

Auth: set an ElectricityMaps API token in ``$ELECTRICITYMAPS_TOKEN`` (or pass
``--token``). Get one from a free account at https://portal.electricitymaps.com
(free tier ~5 zones); historical ``past-range`` access needs a plan that includes
history — the academic programme (https://www.electricitymaps.com/research)
grants it. A free token typically only serves ``latest`` / recent ``history``,
so ``past-range`` may return 401/403 until the plan includes historical data;
the script reports that per zone rather than failing silently.

Five of the 16 are sub-zones on EM (no single national feed); ``ZONE_MAP`` picks
a representative matching the proposal's intent. Verify ids against the free
``GET /v3/zones`` list (``--list-zones``) and override with ``--map`` if needed.

Usage::

    export ELECTRICITYMAPS_TOKEN=...        # free/academic token
    python -m experiments.fetch_electricitymaps_traces \\
        --start 2024-07-01 --end 2024-07-15 --out data_cache/real_traces
    # one zone / sanity check:
    python -m experiments.fetch_electricitymaps_traces --zones DE --start 2024-07-01 --end 2024-07-03
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_BASE_URL = "https://api.electricitymap.org/v3"

# Our 16 zone ids → the ElectricityMaps zone id to download. Most are 1:1; the
# five sub-zoned countries use a representative sub-zone (edit via --map). These
# follow the modelling intent documented in carbon_trace._GRID_ZONES.
ZONE_MAP: dict[str, str] = {
    "DE": "DE",
    "US-CA": "US-CAL-CISO",   # California ISO
    "FR": "FR",
    "PL": "PL",
    "VN": "VN",
    "JP": "JP-TK",            # Tokyo (dominant load centre)
    "GB": "GB",
    "SG": "SG",
    "KR": "KR",
    "BR": "BR-CS",            # Brazil Central-South (~70% of national load)
    "NO": "NO-NO1",           # Oslo region (representative; NO is split NO1..5)
    "ZA": "ZA",
    "AU": "AU-NSW",           # NEM, New South Wales (representative)
    "IN": "IN-WE",            # India West (large industrial region)
    "CN": "CN",
    "AE": "AE",
    # Cross-zone coverage expansion (geographic + carbon-profile diversity).
    "US-TEX": "US-TEX-ERCO",  # ERCOT (Texas): wind + solar + gas, large swing
    "US-PJM": "US-MIDA-PJM",  # PJM mid-Atlantic: coal + gas + nuclear
    "CA-ON": "CA-ON",         # Ontario: nuclear + hydro, clean and flat
    "ES": "ES",               # Spain: solar + wind, large diurnal swing
    "IT": "IT",               # Italy: gas-heavy, mid intensity
    "NL": "NL",               # Netherlands: gas + wind
    "SE": "SE",               # Sweden: hydro + nuclear, very clean
    "DK": "DK-DK1",           # Denmark West: wind, very swingy/intermittent
    "IE": "IE",               # Ireland: wind + gas
    "TW": "TW",               # Taiwan: coal + gas, dirty
    "TH": "TH",               # Thailand: gas, flat baseload
    "ID": "ID",               # Indonesia: coal-dominated, dirty
    "MX": "MX",               # Mexico: gas, mid intensity
    "CL": "CL-SEN",           # Chile (SEN): solar + coal, swingy
}


def _request(url: str, token: str | None) -> dict:
    req = urllib.request.Request(url)
    if token:
        req.add_header("auth-token", token)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read())


def list_zones(base_url: str) -> dict:
    """GET /v3/zones (no auth) — the catalogue of valid EM zone ids."""
    return _request(f"{base_url}/zones", token=None)


def _past_range_one(
    base_url: str, token: str, em_zone: str, start: datetime, end: datetime,
) -> list[tuple[datetime, float]]:
    """One past-range request (the endpoint caps the window length, so callers chunk)."""
    params = {
        "zone": em_zone,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    url = f"{base_url}/carbon-intensity/past-range?{urllib.parse.urlencode(params)}"
    data = _request(url, token)
    rows: list[tuple[datetime, float]] = []
    for item in data.get("data", []):
        ci = item.get("carbonIntensity")
        raw_t = item.get("datetime")
        if ci is None or raw_t is None:
            continue
        t = datetime.fromisoformat(raw_t.replace("Z", "+00:00")).astimezone(UTC)
        rows.append((t, float(ci)))
    return rows


def fetch_past_range(
    base_url: str, token: str, em_zone: str, start: datetime, end: datetime,
    chunk_days: int = 10,
) -> list[tuple[datetime, float]]:
    """Hourly (datetime, carbonIntensity gCO2eq/kWh LCA) for one EM zone over [start, end).

    The past-range endpoint rejects windows longer than ~10 days (HTTP 400), so
    the range is fetched in ``chunk_days`` slices and concatenated (deduped by
    timestamp at the chunk boundaries).
    """
    from datetime import timedelta

    merged: dict[datetime, float] = {}
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        for t, v in _past_range_one(base_url, token, em_zone, cur, nxt):
            merged[t] = v
        cur = nxt
    return sorted(merged.items())


def write_csv(path: Path, zone_id: str, em_zone: str, rows: list[tuple[datetime, float]]) -> None:
    """Write the EM-compatible schema ``load_electricitymaps_csv`` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "Datetime (UTC)", "Country", "Zone Name", "Zone Id",
            "Carbon Intensity gCO₂eq/kWh (LCA)",
        ])
        for ts, val in rows:
            w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), em_zone, em_zone, zone_id, f"{val:.2f}"])


def run(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")

    if args.list_zones:
        zones = list_zones(base_url)
        ids = sorted(zones.get("zones", zones).keys()) if isinstance(zones, dict) else []
        print(f"{len(ids)} EM zones available; ours map to:")
        for k, v in ZONE_MAP.items():
            mark = "ok" if v in ids else "CHECK"
            print(f"  {k:6s} -> {v:14s} [{mark}]")
        return 0

    token = args.token or os.environ.get("ELECTRICITYMAPS_TOKEN")
    if not token:
        print("ERROR: no token. Set $ELECTRICITYMAPS_TOKEN or pass --token "
              "(free account at https://portal.electricitymaps.com).")
        return 2

    if args.map:
        for pair in args.map:
            k, _, v = pair.partition("=")
            if k in ZONE_MAP and v:
                ZONE_MAP[k] = v

    zones = args.zones or list(ZONE_MAP)
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    out_dir = Path(args.out)
    print(f"Fetching {(end - start).days}-day window {args.start} → {args.end} "
          f"for {len(zones)} zones from {base_url}")

    ok, failed = 0, []
    for i, zone in enumerate(zones):
        if i > 0:
            time.sleep(args.rate_limit_sleep_s)
        em_zone = ZONE_MAP.get(zone, zone)
        print(f"  {zone:6s} ({em_zone}): …", end="", flush=True)
        try:
            rows = fetch_past_range(base_url, token, em_zone, start, end, chunk_days=args.chunk_days)
        except urllib.error.HTTPError as e:
            hint = ""
            if e.code in (401, 403):
                hint = " — token lacks historical/this-zone access; needs a plan with past data (academic programme)"
            print(f" FAIL ({e.code}){hint}")
            failed.append((zone, e.code))
            continue
        except urllib.error.URLError as e:
            print(f" FAIL (network: {e.reason})")
            failed.append((zone, "net"))
            continue
        if not rows:
            print(" empty (no data returned)")
            failed.append((zone, "empty"))
            continue
        target = out_dir / f"{zone.lower()}_{args.start}_{args.end}_hourly.csv"
        write_csv(target, zone, em_zone, rows)
        print(f" wrote {len(rows)} rows → {target}")
        ok += 1

    print(f"\nDone: {ok}/{len(zones)} zones written; {len(failed)} failed: {failed}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zones", nargs="+", default=None,
                   help="Our zone ids to fetch (default: all 16 in ZONE_MAP).")
    p.add_argument("--start", default="2024-07-01", help="Inclusive start date (UTC).")
    p.add_argument("--end", default="2024-07-15", help="Exclusive end date (UTC).")
    p.add_argument("--out", default="data_cache/real_traces")
    p.add_argument("--token", default=None, help="EM API token (else $ELECTRICITYMAPS_TOKEN).")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="EM API base; newer tokens may use https://api-access.electricitymaps.com.")
    p.add_argument("--map", nargs="+", default=None,
                   help="Override zone ids, e.g. --map BR=BR-S AU=AU-QLD.")
    p.add_argument("--list-zones", action="store_true",
                   help="List EM zone ids (no auth) and check the ZONE_MAP; do not fetch.")
    p.add_argument("--chunk-days", type=int, default=10,
                   help="Max days per past-range request; the endpoint rejects ~>10-day windows.")
    p.add_argument("--rate-limit-sleep-s", type=float, default=2.0)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
