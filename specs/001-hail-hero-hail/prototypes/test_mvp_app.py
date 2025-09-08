#!/usr/bin/env python3
"""Playwright smoke test for Hail Hero MVP.

Launches a browser, navigates to the Flask server, verifies the leads list,
and saves a screenshot.

Usage:
  python test_mvp_app.py --url http://127.0.0.1:5001
"""

from playwright.sync_api import sync_playwright
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)


def run(url: str) -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until='networkidle', timeout=30000)
        # Wait for the leads table or a lead card
        try:
            # Wait for the leads container or at least one lead element
            page.wait_for_selector('#leads, div.lead', timeout=10000)
        except Exception as exc:
            print('Leads selector not found:', exc)
            page.screenshot(path=str(DATA_DIR / 'mvp_app_test_failed.png'))
            browser.close()
            return 2

        # Take a screenshot for verification
        out = DATA_DIR / 'mvp_app_test.png'
        page.screenshot(path=str(out), full_page=True)
        print('Saved test screenshot to', out)
        browser.close()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:5001')
    args = parser.parse_args()
    raise SystemExit(run(args.url))
