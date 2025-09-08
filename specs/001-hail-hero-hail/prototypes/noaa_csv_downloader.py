#!/usr/bin/env python3
"""NOAA / NCEI Storm Events CSV downloader and filter for Hail Hero.

Fetches events for a date range (default last 365 days), filters for hail >= 0.5"
and wind >= 60 mph for Wisconsin and northern Illinois (lat >= 41.5), and writes
raw JSON, a filtered CSV, and provenance JSON to the data folder.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import csv


DEFAULT_HAIL_MIN = 0.5  # inches
DEFAULT_WIND_MIN = 60  # mph
NORTHERN_IL_LAT = 41.5
DATA_DIR = Path(__file__).parent / '..' / 'data'


def ensure_data_dir(base: Path) -> Path:
    d = base.resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_magnitude(rec: Dict[str, Any]) -> Optional[float]:
    """Return numeric magnitude from record, trying several common field names."""
    for key in ('MAGNITUDE', 'MAG', 'magnitude', 'mag'):
        val = rec.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def record_state(rec: Dict[str, Any]) -> Optional[str]:
    """Attempt to extract state string from the record."""
    for key in ('STATE', 'state', 'STATE_FIPS'):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def record_lat(rec: Dict[str, Any]) -> Optional[float]:
    """Attempt to extract a latitude value from the record."""
    for key in ('BEGIN_LAT', 'BEGIN_LATITUDE', 'begin_lat', 'BEGIN_LAT_DECIMAL'):
        v = rec.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def fetch_events(token: Optional[str], start: str, end: str, dataset: str = 'stormevents', limit: int = 1000) -> List[Dict[str, Any]]:
    """Fetch events from NCEI Search API with simple pagination.

    This is a minimal implementation. Production should handle retries, backoff,
    and robust paging.
    """
    base_url = 'https://www.ncei.noaa.gov/access/services/search/v1/data'
    params = {
        'dataset': dataset,
        'startDate': start,
        'endDate': end,
        'limit': limit,
        'format': 'json',
    }

    headers = {'Accept': 'application/json'}
    if token:
        headers['token'] = token

    results: List[Dict[str, Any]] = []
    offset = 1
    while True:
        params['offset'] = offset
        print(f'Fetching offset={offset} limit={limit}')
        resp = requests.get(base_url, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get('results') or data.get('data') or data
        if isinstance(batch, dict):
            batch = batch.get('results', [])
        if not batch:
            break
        if isinstance(batch, list):
            results.extend(batch)
        else:
            break
        if len(batch) < limit:
            break
        offset += limit

    return results


def filter_events(events: List[Dict[str, Any]], hail_min: float, wind_min: float) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for rec in events:
        etype = (rec.get('EVENT_TYPE') or rec.get('eventType') or '').lower()
        mag = get_magnitude(rec)
        state = (rec.get('STATE') or rec.get('state') or '')
        lat = record_lat(rec)

        is_wi = state.upper() in ('WISCONSIN', 'WI') or state == '55' or state.upper().startswith('WI')
        is_il = state.upper() in ('ILLINOIS', 'IL') or state == '17' or state.upper().startswith('IL')

        if is_il and lat is not None and lat < NORTHERN_IL_LAT:
            is_il = False

        if 'hail' in etype:
            if mag is not None and mag >= hail_min and (is_wi or is_il):
                matches.append(rec)
            continue

        if 'wind' in etype:
            if mag is not None and mag >= wind_min and (is_wi or is_il):
                matches.append(rec)
            continue

    return matches


def write_outputs(events: List[Dict[str, Any]], filtered: List[Dict[str, Any]], outdir: Path, start: str, end: str) -> None:
    ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    raw_file = outdir / f'ncei_raw_{start}_{end}_{ts}.json'
    filt_file = outdir / f'ncei_filtered_{start}_{end}_{ts}.csv'
    prov_file = outdir / f'ncei_provenance_{start}_{end}_{ts}.json'

    with raw_file.open('w', encoding='utf-8') as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    keys = set()
    for r in filtered:
        keys.update(r.keys())
    keys = sorted(keys)
    with filt_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for r in filtered:
            writer.writerow({
                k: (json.dumps(r[k]) if isinstance(r.get(k), (dict, list)) else r.get(k))
                for k in keys
            })

    provenance = {
        'created_ts': ts,
        'count_raw': len(events),
        'count_filtered': len(filtered),
        'notes': 'Filtered for hail >= 0.5 in or wind >= 60 mph in WI or northern IL (lat >= 41.5).',
    }
    with prov_file.open('w', encoding='utf-8') as f:
        json.dump(provenance, f, indent=2)

    print('Wrote:')
    print('  ', raw_file)
    print('  ', filt_file)
    print('  ', prov_file)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Automated NOAA CSV downloader and filter for Hail Hero')
    today = datetime.date.today()
    default_end = today.isoformat()
    default_start = (today - datetime.timedelta(days=365)).isoformat()
    parser.add_argument('--start', default=default_start)
    parser.add_argument('--end', default=default_end)
    parser.add_argument('--hail-min', type=float, default=DEFAULT_HAIL_MIN)
    parser.add_argument('--wind-min', type=float, default=DEFAULT_WIND_MIN)
    parser.add_argument('--outdir', default=str(ensure_data_dir(DATA_DIR)))
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args(argv)

    token = os.environ.get('NCEI_TOKEN')
    if not token:
        print('Warning: NCEI_TOKEN not set — attempting unauthenticated requests which may be rate-limited or fail.')

    events = fetch_events(token, args.start, args.end, limit=args.limit)
    print(f'Fetched {len(events)} raw events (may include multiple event types)')

    filtered = filter_events(events, args.hail_min, args.wind_min)
    print(f'Filtered to {len(filtered)} events matching thresholds and geography')

    outdir = Path(args.outdir)
    write_outputs(events, filtered, outdir, args.start, args.end)


if __name__ == '__main__':
    main()
