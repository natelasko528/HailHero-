#!/usr/bin/env python3
"""
NOAA Storm Events CSV Downloader

Uses Playwright to automate downloading hail event data from NOAA Storm Events website.
Filters for Wisconsin and Illinois counties with hail >= 0.75 inches for the last year.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from playwright.async_api import async_playwright

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NOAADataDownloader:
    """Downloads NOAA storm events data using Playwright automation."""
    
    def __init__(self, download_dir: str = "data"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
    async def download_hail_data(self) -> List[Dict[str, Any]]:
        """Download hail event data for target regions."""
        
        # Calculate date range (last year)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Format dates for NOAA form (mm/dd/yyyy)
        start_date_str = start_date.strftime("%m/%d/%Y")
        end_date_str = end_date.strftime("%m/%d/%Y")
        
        logger.info(f"Downloading hail data from {start_date_str} to {end_date_str}")
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            
            # Enable downloads
            await context.route('**/*', lambda route: route.continue_())
            
            page = await context.new_page()
            
            try:
                # Navigate to NOAA Storm Events
                logger.info("Navigating to NOAA Storm Events...")
                await page.goto('https://www.ncei.noaa.gov/stormevents/')
                await page.wait_for_load_state('networkidle')
                
                # Wait for form elements to be ready
                logger.info("Waiting for form to load...")
                await page.wait_for_selector('select[name="statefips"]', timeout=60000)
                
                # Debug: take screenshot to see page state
                await page.screenshot(path="noaa_page_debug.png")
                logger.info("Took screenshot for debugging")
                
                # Get available options
                options = await page.eval_on_selector_all('select[name="statefips"] option', 'els => els.map(el => el.textContent)')
                logger.info(f"Available state options: {options[:10]}...")  # First 10 options
                
                await page.wait_for_selector('input[name="beginDate"]', timeout=30000)
                await page.wait_for_selector('input[name="hailfilter"]', timeout=30000)
                
                # Fill in the search form
                logger.info("Filling in search form...")
                
                # Select Wisconsin (state FIPS 55)
                await page.select_option('select[name="statefips"]', value='55', timeout=30000)
                
                # Set date range
                await page.fill('input[name="beginDate"]', start_date_str)
                await page.fill('input[name="endDate"]', end_date_str)
                
                # Set hail size filter (0.75+ inches)
                await page.fill('input[name="hailfilter"]', '0.75')
                
                # Submit the form
                logger.info("Submitting search form...")
                await page.click('input[type="submit"]')
                await page.wait_for_load_state('networkidle')
                
                # Look for CSV download link
                logger.info("Looking for CSV download link...")
                
                # Try multiple selectors for the download link
                download_selectors = [
                    'a[href*="bulk.csv"]',
                    'a:has-text("Bulk Data Download")',
                    'a:has-text("CSV")',
                    'a[href*=".csv"]'
                ]
                
                download_link = None
                for selector in download_selectors:
                    try:
                        download_link = await page.wait_for_selector(selector, timeout=5000)
                        if download_link:
                            break
                    except:
                        continue
                
                if not download_link:
                    logger.error("Could not find CSV download link")
                    return []
                
                # Start download
                logger.info("Starting CSV download...")
                async with page.expect_download() as download_info:
                    await download_link.click()
                
                download = await download_info
                
                # Save the file
                csv_path = self.download_dir / f"noaa_hail_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                await download.save_as(csv_path)
                
                logger.info(f"Downloaded CSV to: {csv_path}")
                
                # Parse the CSV data
                events = await self._parse_csv_data(csv_path)
                
                # Also download Illinois data (Lake and McHenry counties)
                logger.info("Downloading Illinois data...")
                illinois_events = await self._download_illinois_data(page, start_date_str, end_date_str)
                events.extend(illinois_events)
                
                return events
                
            except Exception as e:
                logger.error(f"Error downloading data: {e}")
                return []
            
            finally:
                await browser.close()
    
    async def _download_illinois_data(self, page, start_date_str: str, end_date_str: str) -> List[Dict[str, Any]]:
        """Download data for Illinois counties."""
        try:
            # Go back to search page
            await page.goto('https://www.ncei.noaa.gov/stormevents/')
            await page.wait_for_load_state('networkidle')
            
            # Select Illinois (state FIPS 17)
            await page.select_option('select[name="statefips"]', value='17')
            
            # Set date range
            await page.fill('input[name="beginDate"]', start_date_str)
            await page.fill('input[name="endDate"]', end_date_str)
            
            # Set hail size filter
            await page.fill('input[name="hailfilter"]', '0.75')
            
            # Submit form
            await page.click('input[type="submit"]')
            await page.wait_for_load_state('networkidle')
            
            # Download CSV
            download_selectors = [
                'a[href*="bulk.csv"]',
                'a:has-text("Bulk Data Download")',
                'a:has-text("CSV")',
                'a[href*=".csv"]'
            ]
            
            download_link = None
            for selector in download_selectors:
                try:
                    download_link = await page.wait_for_selector(selector, timeout=5000)
                    if download_link:
                        break
                except:
                    continue
            
            if download_link:
                async with page.expect_download() as download_info:
                    await download_link.click()
                
                download = await download_info
                
                csv_path = self.download_dir / f"noaa_hail_events_illinois_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                await download.save_as(csv_path)
                
                logger.info(f"Downloaded Illinois CSV to: {csv_path}")
                
                # Parse Illinois data
                illinois_events = await self._parse_csv_data(csv_path)
                
                # Filter for Lake and McHenry counties
                target_counties = ['LAKE', 'MCHENRY']
                filtered_events = [
                    event for event in illinois_events 
                    if event.get('CZ_NAME', '').upper() in target_counties
                ]
                
                logger.info(f"Filtered {len(illinois_events)} Illinois events to {len(filtered_events)} in target counties")
                return filtered_events
            
            return []
            
        except Exception as e:
            logger.error(f"Error downloading Illinois data: {e}")
            return []
    
    async def _parse_csv_data(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Parse CSV data and extract hail events."""
        try:
            import csv
            
            events = []
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Filter for hail events
                    event_type = row.get('EVENT_TYPE', '').upper()
                    if 'HAIL' not in event_type:
                        continue
                    
                    # Extract relevant data
                    event = {
                        'event_id': row.get('EVENT_ID', ''),
                        'event_type': event_type,
                        'begin_date': row.get('BEGIN_DATE', ''),
                        'begin_time': row.get('BEGIN_TIME', ''),
                        'end_date': row.get('END_DATE', ''),
                        'end_time': row.get('END_TIME', ''),
                        'state': row.get('STATE', ''),
                        'county': row.get('CZ_NAME', ''),
                        'magnitude': float(row.get('MAGNITUDE', 0)) if row.get('MAGNITUDE') else 0,
                        'latitude': float(row.get('BEGIN_LAT', 0)) if row.get('BEGIN_LAT') else 0,
                        'longitude': float(row.get('BEGIN_LON', 0)) if row.get('BEGIN_LON') else 0,
                        'injuries_direct': int(row.get('INJURIES_DIRECT', 0)) if row.get('INJURIES_DIRECT') else 0,
                        'deaths_direct': int(row.get('DEATHS_DIRECT', 0)) if row.get('DEATHS_DIRECT') else 0,
                        'damage_property': row.get('DAMAGE_PROPERTY', ''),
                        'source': row.get('SOURCE', ''),
                        'data_source': 'NOAA Storm Events CSV'
                    }
                    
                    # Only include events with hail size >= 0.75 inches
                    if event['magnitude'] >= 0.75:
                        events.append(event)
            
            logger.info(f"Parsed {len(events)} hail events from {csv_path}")
            return events
            
        except Exception as e:
            logger.error(f"Error parsing CSV {csv_path}: {e}")
            return []

async def main():
    """Main function to download NOAA data."""
    downloader = NOAADataDownloader()
    events = await downloader.download_hail_data()
    
    if events:
        logger.info(f"Successfully downloaded {len(events)} hail events")
        
        # Save processed data
        import json
        output_path = Path("data/noaa_hail_events.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(events, f, indent=2)
        
        logger.info(f"Saved processed data to {output_path}")
    else:
        logger.error("No events downloaded")

if __name__ == "__main__":
    asyncio.run(main())