#!/usr/bin/env python3
"""Playwright-based NOAA Storm Events browser automation.

This script attempts to open the NOAA Storm Events query UI, set a date range
and state filters, run a search, and download the CSV results. The NOAA UI can
change; this prototype tries several heuristics and saves diagnostic screenshots
if an automated download cannot be completed.

Usage:
  python noaa_playwright_downloader.py --start 2024-09-08 --end 2025-09-08 --headless

Note: requires `playwright` Python package and `playwright install` browser
binaries. The script will attempt to save downloaded CSVs to the spec data
folder: specs/001-hail-hero-hail/data/
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

from playwright.sync_api import (
    sync_playwright,
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
)


DATA_DIR = Path(__file__).parent / '..' / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

NOAA_URL = 'https://www.ncdc.noaa.gov/stormevents/'


def run(start: str, end: str, headless: bool = True) -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        page.goto(NOAA_URL, wait_until='load', timeout=60000)
        time.sleep(2)
        page.screenshot(path=str(DATA_DIR / 'noaa_page_initial.png'))

        # Attempt to set date range using common input selectors
        try:
            # Try inputs named 'startDate'/'endDate' or type=date inputs
            page.evaluate(
                "(args) => {"
                "  const start = document.querySelector('input[name=startDate]') || document.querySelector('input[type=date]');"
                "  const end = document.querySelector('input[name=endDate]') || Array.from(document.querySelectorAll('input[type=date]'))[1];"
                "  if (start) start.value = args.s;"
                "  if (end) end.value = args.e;"
                "}",
                {"s": start, "e": end},
            )
        except PlaywrightError as exc:
            print('Date-set heuristic failed:', exc)

        # Heuristic: try to open the state selector and choose Wisconsin + Illinois
        try:
            # Try to find a state filter text input or dropdown and set value
            page.fill('input[placeholder="Select state"]', 'Wisconsin')
            time.sleep(0.5)
            page.keyboard.press('Enter')
            page.fill('input[placeholder="Select state"]', 'Illinois')
            time.sleep(0.5)
            page.keyboard.press('Enter')
        except PlaywrightError as exc:
            # If specific selectors not present or interaction fails, log and continue
            print('State selector heuristic skipped:', exc)

        # Attempt to click any 'Search' or 'Apply' buttons
        for label in ('Search', 'Apply', 'Submit', 'Filter'):
            try:
                btn = page.query_selector(f'text="{label}"')
                if btn:
                    try:
                        btn.click()
                        print(f'Clicked button: {label}')
                        break
                    except PlaywrightTimeoutError as exc:
                        print(f'Click timed out for {label}:', exc)
                        continue
            except PlaywrightError as exc:
                # Log selector/query errors and continue trying other labels
                print(f'Error finding/clicking button {label}:', exc)
                continue

        time.sleep(3)
        page.screenshot(path=str(DATA_DIR / 'noaa_after_search.png'))

        # Attempt to click a CSV/Download link
        download_paths = []
        # The page may contain a link text like 'CSV' or 'Download'
        for dl_text in ('CSV', 'Download', 'Export'):
            try:
                with page.expect_download(timeout=5000) as download_info:
                    el = page.query_selector(f'text="{dl_text}"')
                    if el:
                        try:
                            el.click()
                            download = download_info.value
                            path = DATA_DIR / download.suggested_filename
                            download.save_as(str(path))
                            download_paths.append(path)
                            print('Downloaded:', path)
                        except PlaywrightTimeoutError as exc:
                            print(f'Download timed out for {dl_text}:', exc)
                            continue
            except (PlaywrightTimeoutError, PlaywrightError):
                # No download triggered for this label; try next
                continue

        # If no downloads, save a final diagnostic screenshot
        if not download_paths:
            print('No downloads completed automatically; capturing diagnostic artifacts.')
            page.screenshot(path=str(DATA_DIR / 'noaa_no_download.png'))

        context.close()
        browser.close()

    return 0 if download_paths else 2


def iso_date(d: str) -> str:
    # Basic ISO date validation
    datetime.datetime.fromisoformat(d)
    return d


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    today = datetime.date.today()
    default_end = today.isoformat()
    default_start = (today - datetime.timedelta(days=365)).isoformat()
    parser.add_argument('--start', default=default_start, type=iso_date)
    parser.add_argument('--end', default=default_end, type=iso_date)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args(argv)

    return run(args.start, args.end, headless=args.headless)


if __name__ == '__main__':
    raise SystemExit(main())
