"""Fetch marginal carbon-intensity (MOER) traces from the WattTime API.

The headline real-trace results replay *average* grid intensity (ElectricityMaps
LCA). The unresolved attribution question is whether added training demand should
be charged at grid-*average* or grid-*marginal* generation intensity. WattTime's
Marginal Operating Emissions Rate (MOER) is the marginal signal; this script
pulls historical MOER per zone and writes it under the *same* CSV schema the
harness already reads, so ``exp_realtrace_pareto`` can replay over marginal
traces with no logic change — point it at a parallel directory:

    python -m experiments.exp_realtrace_pareto --real-dir data_cache/marginal_traces ...

Marginal data is genuinely separate from the average traces, so the two live in
sibling directories (``data_cache/real_traces`` = average, ``marginal_traces`` =
marginal) and the per-zone avg-vs-marginal contrast is a paired comparison.

Coverage / data class. WattTime's native, highest-fidelity MOER is North America
(CAISO, ERCOT, PJM, IESO, ...). International regions are available on the
research/enterprise tiers at lower fidelity (modelled rather than balancing-
authority-metered). The script resolves every zone through WattTime's own
``region-from-loc`` endpoint (the version-proof, recommended path) from a
representative load-centre lat/lon, records the resolved region id, and annotates
each zone NA-native vs international so the writeup can report the data class.

Auth (WattTime v3, https://docs.watttime.org). One-time registration is a POST to
``/register`` (or the research-access form at watttime.org/data-science/for-research,
which must be approved before historical access is granted). At run time:

    export WATTTIME_USERNAME=...    # and
    export WATTTIME_PASSWORD=...    # -> the script logs in for a ~30-min bearer token
    # or, if you already minted one:
    export WATTTIME_TOKEN=...

Login is ``GET /login`` with HTTP Basic auth returning ``{"token": ...}``; data
calls then send ``Authorization: Bearer <token>``.

Unit conversion. MOER is reported in lb CO2 / MWh. We convert to gCO2eq/kWh with
``lb/MWh x 0.45359237 = g/kWh`` (1 lb = 453.59237 g; /1000 for MWh->kWh) and write
the ``intensity_g_per_kwh`` column, which the loader recognises; the raw
``moer_lbs_per_mwh`` value and the resolved region/signal are kept as provenance
columns the loader ignores.

Fallback if WattTime access is denied or a zone is uncovered: Singularity
(US, 60-day historical) or GridStatus (ISO public feeds) for the four NA zones.

Usage::

    python -m experiments.fetch_watttime_traces --list-regions   # resolve+print, no fetch (needs login)
    python -m experiments.fetch_watttime_traces \\
        --start 2024-07-01 --end 2024-07-15 --out data_cache/marginal_traces
    python -m experiments.fetch_watttime_traces \\
        --start 2025-01-01 --end 2025-01-15 --out data_cache/marginal_traces_winter
    # one zone, force a known balancing authority:
    python -m experiments.fetch_watttime_traces --zones US-CA --region US-CA=CAISO_NORTH
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_BASE_URL = "https://api.watttime.org"
SIGNAL_TYPE = "co2_moer"
LB_PER_MWH_TO_G_PER_KWH = 0.45359237  # 1 lb = 453.59237 g; MWh -> kWh divides by 1000.

# Representative load-centre coordinates per zone id. region-from-loc resolves
# each to a WattTime region, so we never hard-code (version-dependent) BA codes
# as the default. Mirrors the zone set in fetch_electricitymaps_traces.ZONE_MAP.
ZONE_LATLON: dict[str, tuple[float, float]] = {
    "DE": (52.52, 13.40),       # Berlin
    "US-CA": (37.77, -122.42),  # San Francisco (CAISO north)
    "FR": (48.85, 2.35),        # Paris
    "PL": (52.23, 21.01),       # Warsaw
    "VN": (21.03, 105.85),      # Hanoi
    "JP": (35.68, 139.69),      # Tokyo
    "GB": (51.51, -0.13),       # London
    "SG": (1.35, 103.82),       # Singapore
    "KR": (37.57, 126.98),      # Seoul
    "BR": (-23.55, -46.63),     # Sao Paulo (BR central-south)
    "NO": (59.91, 10.75),       # Oslo (NO1)
    "ZA": (-26.20, 28.05),      # Johannesburg
    "AU": (-33.87, 151.21),     # Sydney (NEM NSW)
    "IN": (19.08, 72.88),       # Mumbai (India west)
    "CN": (39.90, 116.40),      # Beijing
    "AE": (24.45, 54.38),       # Abu Dhabi
    "US-TEX": (29.76, -95.37),  # Houston (ERCOT)
    "US-PJM": (39.95, -75.17),  # Philadelphia (PJM mid-Atlantic)
    "CA-ON": (43.65, -79.38),   # Toronto (IESO)
    "ES": (40.42, -3.70),       # Madrid
    "IT": (41.90, 12.50),       # Rome
    "NL": (52.37, 4.90),        # Amsterdam
    "SE": (59.33, 18.07),       # Stockholm
    "DK": (56.16, 10.20),       # Aarhus (DK1, west Denmark)
    "IE": (53.35, -6.26),       # Dublin
    "TW": (25.03, 121.57),      # Taipei
    "TH": (13.76, 100.50),      # Bangkok
    "ID": (-6.21, 106.85),      # Jakarta
    "MX": (19.43, -99.13),      # Mexico City
    "CL": (-33.45, -70.67),     # Santiago (SEN)
}

# WattTime's native MOER balancing authorities (highest fidelity). The rest are
# international (research/enterprise tier, modelled). Used only to annotate the
# per-zone data class in the output, not to drive resolution.
NA_NATIVE_ZONES = {"US-CA", "US-TEX", "US-PJM", "CA-ON"}


def _request(url: str, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url)
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read())


def login(base_url: str, username: str, password: str) -> str:
    """GET /login with HTTP Basic auth -> a ~30-minute bearer token."""
    creds = base64.b64encode(f"{username}:{password}".encode()).decode()
    data = _request(f"{base_url}/login", {"Authorization": f"Basic {creds}"})
    token = data.get("token")
    if not token:
        raise RuntimeError(f"login returned no token: {data}")
    return token


def region_from_loc(base_url: str, token: str, lat: float, lon: float) -> dict:
    """Resolve a lat/lon to the WattTime region serving co2_moer there."""
    params = {"latitude": lat, "longitude": lon, "signal_type": SIGNAL_TYPE}
    url = f"{base_url}/v3/region-from-loc?{urllib.parse.urlencode(params)}"
    return _request(url, {"Authorization": f"Bearer {token}"})


def _historical_one(
    base_url: str, token: str, region: str, start: datetime, end: datetime,
) -> list[tuple[datetime, float]]:
    """One /v3/historical request (the endpoint caps the window, so callers chunk)."""
    params = {
        "region": region,
        "start": start.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "signal_type": SIGNAL_TYPE,
    }
    url = f"{base_url}/v3/historical?{urllib.parse.urlencode(params)}"
    data = _request(url, {"Authorization": f"Bearer {token}"})
    rows: list[tuple[datetime, float]] = []
    for item in data.get("data", []):
        val = item.get("value")
        raw_t = item.get("point_time") or item.get("datetime")
        if val is None or raw_t is None:
            continue
        t = datetime.fromisoformat(raw_t.replace("Z", "+00:00")).astimezone(UTC)
        rows.append((t, float(val)))
    return rows


def fetch_historical(
    base_url: str, token: str, region: str, start: datetime, end: datetime,
    chunk_days: int = 30,
) -> list[tuple[datetime, float]]:
    """MOER (datetime, lb/MWh) for one region over [start, end), chunked + deduped.

    The historical endpoint rejects windows longer than ~32 days, so the range is
    fetched in ``chunk_days`` slices (default 30, a safe margin) and merged.
    """
    merged: dict[datetime, float] = {}
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        for t, v in _historical_one(base_url, token, region, cur, nxt):
            merged[t] = v
        cur = nxt
    return sorted(merged.items())


def write_csv(
    path: Path, zone_id: str, region: str, rows: list[tuple[datetime, float]],
) -> None:
    """Write the drop-in schema: Datetime (UTC) + intensity_g_per_kwh (loader-recognised),
    plus MOER provenance columns the loader ignores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "Datetime (UTC)", "intensity_g_per_kwh",
            "signal_type", "region", "moer_lbs_per_mwh",
        ])
        for ts, moer in rows:
            g_per_kwh = moer * LB_PER_MWH_TO_G_PER_KWH
            w.writerow([
                ts.strftime("%Y-%m-%d %H:%M:%S"), f"{g_per_kwh:.2f}",
                SIGNAL_TYPE, region, f"{moer:.2f}",
            ])


def _resolve_regions(
    base_url: str, token: str, zones: list[str], overrides: dict[str, str],
) -> dict[str, dict]:
    """Resolve each zone to a region (override else region-from-loc), with data class."""
    resolved: dict[str, dict] = {}
    for zone in zones:
        klass = "marginal-na-native" if zone in NA_NATIVE_ZONES else "marginal-international"
        if zone in overrides:
            resolved[zone] = {"region": overrides[zone], "via": "override", "data_class": klass}
            continue
        lat, lon = ZONE_LATLON.get(zone, (None, None))
        if lat is None:
            resolved[zone] = {"region": None, "via": "no-latlon", "data_class": klass}
            continue
        try:
            r = region_from_loc(base_url, token, lat, lon)
            resolved[zone] = {
                "region": r.get("region"),
                "region_full_name": r.get("region_full_name"),
                "via": "region-from-loc",
                "data_class": klass,
            }
        except urllib.error.HTTPError as e:
            resolved[zone] = {"region": None, "via": f"resolve-fail-{e.code}", "data_class": klass}
    return resolved


def run(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")

    overrides: dict[str, str] = {}
    if args.region:
        for pair in args.region:
            k, _, v = pair.partition("=")
            if k and v:
                overrides[k] = v

    token = args.token or os.environ.get("WATTTIME_TOKEN")
    if not token:
        username = args.username or os.environ.get("WATTTIME_USERNAME")
        password = args.password or os.environ.get("WATTTIME_PASSWORD")
        if not (username and password):
            print("ERROR: no credentials. Set $WATTTIME_TOKEN, or $WATTTIME_USERNAME + "
                  "$WATTTIME_PASSWORD (research access must be approved for historical data).")
            return 2
        try:
            token = login(base_url, username, password)
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as e:
            print(f"ERROR: login failed: {e}")
            return 2

    zones = args.zones or list(ZONE_LATLON)
    resolved = _resolve_regions(base_url, token, zones, overrides)

    if args.list_regions:
        print(f"Resolved {len(zones)} zones -> WattTime regions ({SIGNAL_TYPE}):")
        for z in zones:
            info = resolved[z]
            reg = info.get("region") or "—"
            print(f"  {z:7s} -> {reg:18s} [{info['via']}, {info['data_class']}]")
        return 0

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    out_dir = Path(args.out)
    print(f"Fetching {(end - start).days}-day MOER window {args.start} -> {args.end} "
          f"for {len(zones)} zones from {base_url}")

    ok, failed = 0, []
    for i, zone in enumerate(zones):
        if i > 0:
            time.sleep(args.rate_limit_sleep_s)
        region = resolved[zone].get("region")
        if not region:
            print(f"  {zone:7s}: SKIP (no region; {resolved[zone]['via']})")
            failed.append((zone, "no-region"))
            continue
        print(f"  {zone:7s} ({region}): …", end="", flush=True)
        try:
            rows = fetch_historical(base_url, token, region, start, end, chunk_days=args.chunk_days)
        except urllib.error.HTTPError as e:
            hint = " — token lacks historical/this-region access (needs approved research plan)" \
                if e.code in (401, 403) else ""
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
        target = out_dir / f"{zone.lower()}_{args.start}_{args.end}_moer.csv"
        write_csv(target, zone, region, rows)
        print(f" wrote {len(rows)} rows -> {target}")
        ok += 1

    meta = out_dir / "_resolved_regions.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(resolved, indent=2))
    print(f"\nDone: {ok}/{len(zones)} zones written; {len(failed)} failed: {failed}")
    print(f"Resolved-region map -> {meta}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zones", nargs="+", default=None,
                   help="Zone ids to fetch (default: all in ZONE_LATLON).")
    p.add_argument("--start", default="2024-07-01", help="Inclusive start date (UTC).")
    p.add_argument("--end", default="2024-07-15", help="Exclusive end date (UTC).")
    p.add_argument("--out", default="data_cache/marginal_traces")
    p.add_argument("--token", default=None, help="Pre-minted bearer token (else login via creds).")
    p.add_argument("--username", default=None, help="WattTime username (else $WATTTIME_USERNAME).")
    p.add_argument("--password", default=None, help="WattTime password (else $WATTTIME_PASSWORD).")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--region", nargs="+", default=None,
                   help="Force a region, e.g. --region US-CA=CAISO_NORTH US-TEX=ERCOT.")
    p.add_argument("--list-regions", action="store_true",
                   help="Resolve and print zone->region (needs login); do not fetch.")
    p.add_argument("--chunk-days", type=int, default=30,
                   help="Max days per historical request; the endpoint rejects ~>32-day windows.")
    p.add_argument("--rate-limit-sleep-s", type=float, default=2.0)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
