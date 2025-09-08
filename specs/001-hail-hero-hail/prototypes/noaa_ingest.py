#!/usr/bin/env python3
"""NOAA / NCEI Storm Events ingestion prototype.

Small script to fetch NCEI storm event data and write a JSON output. This
prototype demonstrates the API call pattern and requires an NCEI token for
most endpoints. Production code should add paging, retries, and robust error
handling.
"""

import os
import argparse
import datetime
import json
from pathlib import Path
from typing import Any

import requests


def ensure_output_dir(base: Path) -> Path:
    """Ensure the output directory exists and return it.

    Args:
        base: Base path where a `data` directory will be created.
    """
    out = base / "data"
    out.mkdir(parents=True, exist_ok=True)
    return out


def fetch_ncei_events(token: str, start: str, end: str, dataset: str = "stormevents", location: str | None = None, limit: int = 1000) -> Any:
    """Fetch events from the NCEI API.

    Note: This is a minimal wrapper for demonstration. It returns the parsed
    JSON response. Production code should implement paging, retries, and rate
    limit handling.
    """
    base_url = "https://www.ncei.noaa.gov/access/services/search/v1/data"
    params = {
        'dataset': dataset,
        'startDate': start,
        'endDate': end,
        'limit': limit,
        'format': 'json',
    }

    if location:
        # Allow shorthand like 'state:TX' to be included in a free-text query
        if location.startswith('state:'):
            params['q'] = location
        else:
            params['bbox'] = location

    headers = {
        'Accept': 'application/json',
    }
    if token:
        headers['token'] = token

    resp = requests.get(base_url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def save_json(obj: Any, path: Path) -> None:
    """Save object as indented JSON to the specified path."""
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def iso_date(d: str) -> str:
    """Validate and normalize ISO date (YYYY-MM-DD)."""
    try:
        dt = datetime.datetime.fromisoformat(d)
        return dt.date().isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date: {d}") from exc


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='NOAA/NCEI Storm Events ingestion prototype')
    parser.add_argument('--start', type=iso_date, required=True)
    parser.add_argument('--end', type=iso_date, required=True)
    parser.add_argument('--dataset', default='stormevents')
    parser.add_argument('--location', default=None, help='Optional location filter; e.g. "state:TX" or bbox as "minlon,minlat,maxlon,maxlat"')
    parser.add_argument('--outdir', default=str(Path(__file__).parent))
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args(argv)

    token = os.environ.get('NCEI_TOKEN')
    if not token:
        print(
            'NCEI_TOKEN not found in environment. To use NOAA/NCEI API, request a token at '
            'https://www.ncei.noaa.gov/support/access-service-api'
        )
        print('This prototype will not attempt unauthenticated requests.')
        return

    out_base = Path(args.outdir)
    out_dir = ensure_output_dir(out_base)

    print(f'Fetching NCEI events {args.start} -> {args.end} (dataset={args.dataset})')
    data = fetch_ncei_events(token, args.start, args.end, dataset=args.dataset, location=args.location, limit=args.limit)

    timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_file = out_dir / f'ncei_events_{args.start}_{args.end}_{timestamp}.json'
    save_json(data, out_file)

    print(f'Wrote {out_file} ({len(json.dumps(data))} bytes)')


if __name__ == '__main__':
    main()
