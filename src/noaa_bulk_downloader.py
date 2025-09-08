#!/usr/bin/env python3
"""
NOAA Storm Events Bulk CSV Downloader

Downloads bulk CSV data directly from NOAA's bulk data endpoint.
Filters for hail events in Wisconsin and Illinois counties.
"""

import asyncio
import logging
import aiohttp
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NOAABulkDownloader:
    """Downloads NOAA storm events data using bulk CSV endpoint."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    async def download_bulk_data(self, year: int = 2024) -> List[Dict[str, Any]]:
        """Download bulk CSV data for a specific year."""
        
        # NOAA bulk CSV URLs for different years
        bulk_urls = {
            2024: "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2024_c20250117.csv.gz",
            2023: "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2023_c20240116.csv.gz",
            2022: "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2022_c20230118.csv.gz",
            2021: "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2021_c20220118.csv.gz"
        }
        
        if year not in bulk_urls:
            logger.error(f"No bulk URL available for year {year}")
            return []
        
        url = bulk_urls[year]
        logger.info(f"Downloading bulk data for {year} from {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        # Save the compressed file
                        compressed_path = self.data_dir / f"stormevents_{year}.csv.gz"
                        with open(compressed_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        
                        logger.info(f"Downloaded compressed data to {compressed_path}")
                        
                        # Extract and process the CSV
                        events = await self._process_compressed_csv(compressed_path)
                        
                        return events
                    else:
                        logger.error(f"Failed to download: HTTP {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error downloading bulk data: {e}")
            return []
    
    async def _process_compressed_csv(self, compressed_path: Path) -> List[Dict[str, Any]]:
        """Process compressed CSV file and extract hail events."""
        try:
            import gzip
            
            events = []
            target_states = ['WISCONSIN', 'ILLINOIS']
            target_counties = ['LAKE', 'MCHENRY']  # For Illinois filtering
            
            logger.info(f"Processing {compressed_path}")
            
            with gzip.open(compressed_path, 'rt', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                
                processed_count = 0
                hail_count = 0
                
                for row in reader:
                    processed_count += 1
                    
                    # Filter for target states
                    state = row.get('STATE', '').upper()
                    if state not in target_states:
                        continue
                    
                    # Filter for hail events only
                    event_type = row.get('EVENT_TYPE', '').upper()
                    if 'HAIL' not in event_type:
                        continue
                    
                    # For Illinois, filter for specific counties
                    if state == 'ILLINOIS':
                        county = row.get('CZ_NAME', '').upper()
                        if county not in target_counties:
                            continue
                    
                    # Parse magnitude (hail size)
                    try:
                        magnitude = float(row.get('MAGNITUDE', 0))
                    except (ValueError, TypeError):
                        magnitude = 0
                    
                    # Only include events with hail size >= 0.75 inches
                    if magnitude < 0.75:
                        continue
                    
                    hail_count += 1
                    
                    # Extract relevant data
                    event = {
                        'event_id': row.get('EVENT_ID', ''),
                        'event_type': event_type,
                        'begin_date': row.get('BEGIN_DATE', ''),
                        'begin_time': row.get('BEGIN_TIME', ''),
                        'end_date': row.get('END_DATE', ''),
                        'end_time': row.get('END_TIME', ''),
                        'state': state,
                        'county': row.get('CZ_NAME', ''),
                        'magnitude': magnitude,
                        'latitude': float(row.get('BEGIN_LAT', 0)) if row.get('BEGIN_LAT') else 0,
                        'longitude': float(row.get('BEGIN_LON', 0)) if row.get('BEGIN_LON') else 0,
                        'injuries_direct': int(row.get('INJURIES_DIRECT', 0)) if row.get('INJURIES_DIRECT') else 0,
                        'deaths_direct': int(row.get('DEATHS_DIRECT', 0)) if row.get('DEATHS_DIRECT') else 0,
                        'damage_property': row.get('DAMAGE_PROPERTY', ''),
                        'source': row.get('SOURCE', ''),
                        'episode_narrative': row.get('EPISODE_NARRATIVE', ''),
                        'event_narrative': row.get('EVENT_NARRATIVE', ''),
                        'data_source': f'NOAA Storm Events Bulk CSV {compressed_path.stem}'
                    }
                    
                    events.append(event)
            
            logger.info(f"Processed {processed_count} records, found {hail_count} hail events, extracted {len(events)} target events")
            return events
            
        except Exception as e:
            logger.error(f"Error processing compressed CSV: {e}")
            return []

async def download_multiple_years() -> List[Dict[str, Any]]:
    """Download data for multiple recent years."""
    downloader = NOAABulkDownloader()
    all_events = []
    
    # Download last 2 years of data
    for year in [2024, 2023]:
        logger.info(f"Downloading data for {year}")
        events = await downloader.download_bulk_data(year)
        all_events.extend(events)
        
        # Save individual year data
        year_path = Path(f"data/noaa_hail_events_{year}.json")
        with open(year_path, 'w') as f:
            json.dump(events, f, indent=2)
        logger.info(f"Saved {year} data to {year_path}")
    
    # Save combined data
    combined_path = Path("data/noaa_hail_events_combined.json")
    with open(combined_path, 'w') as f:
        json.dump(all_events, f, indent=2)
    
    logger.info(f"Saved combined data ({len(all_events)} events) to {combined_path}")
    return all_events

async def main():
    """Main function to download NOAA data."""
    events = await download_multiple_years()
    
    if events:
        logger.info(f"Successfully downloaded {len(events)} hail events")
        
        # Print summary statistics
        states = {}
        for event in events:
            state = event['state']
            states[state] = states.get(state, 0) + 1
        
        logger.info("Events by state:")
        for state, count in states.items():
            logger.info(f"  {state}: {count}")
        
        # Print some example events
        logger.info("Example events:")
        for i, event in enumerate(events[:3]):
            logger.info(f"  {i+1}. {event['county']}, {event['state']} - {event['magnitude']}\" hail on {event['begin_date']}")
    else:
        logger.error("No events downloaded")

if __name__ == "__main__":
    asyncio.run(main())