#!/usr/bin/env python3
"""Enhanced MVP runner: fetch NCEI/NOAA storm events and produce comprehensive lead records.

Behavior:
- If environment variable NCEI_TOKEN is set, fetch events from NCEI Search API.
- Otherwise create synthetic leads for WI/IL for demo purposes.
- Outputs written to `specs/001-hail-hero-hail/data/leads.jsonl` and raw events JSON.
- Enhanced with proper error handling, rate limiting, fallback mechanisms, and property enrichment.
- Implements Hail Recon-style property enrichment with sophisticated lead scoring.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

import requests

# Import address enrichment module
try:
    from .address_enrichment import AddressEnricher, create_enricher
except ImportError:
    # Fallback for direct execution
    from address_enrichment import AddressEnricher, create_enricher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspaces/HailHero-/noaa_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'specs' / '001-hail-hero-hail' / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HAIL_MIN = 0.5
DEFAULT_WIND_MIN = 60
DEFAULT_API_TIMEOUT = 30
DEFAULT_RATE_LIMIT_DELAY = 1.0  # seconds between API calls
MAX_RETRIES = 3

# NOAA API Configuration
NCEI_BASE_URL = 'https://www.ncei.noaa.gov/access/services/search/v1/data'
NCEI_DATASET = 'stormevents'

@dataclass
class APIConfig:
    """Configuration for NOAA API integration."""
    token: Optional[str] = None
    base_url: str = NCEI_BASE_URL
    timeout: int = DEFAULT_API_TIMEOUT
    rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY
    max_retries: int = MAX_RETRIES
    dataset: str = NCEI_DATASET

class DataSource(Enum):
    """Enum for data sources."""
    NOAA_API = "noaa_api"
    SYNTHETIC = "synthetic"
    CACHED = "cached"

@dataclass
class IngestionResult:
    """Result of data ingestion process."""
    source: DataSource
    events_count: int
    leads_count: int
    success: bool
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    enriched_leads_count: int = 0
    quality_score_avg: float = 0.0

@dataclass
class PropertyEnrichmentConfig:
    """Configuration for property enrichment."""
    enable_enrichment: bool = True
    max_properties_per_event: int = 50
    search_radius_miles: float = 5.0
    property_cache_enabled: bool = True
    openaddresses_enabled: bool = True
    nominatim_enabled: bool = True
    rate_limit_delay: float = 1.0
    quality_threshold: float = 60.0

class LeadTier(Enum):
    """Lead quality tiers."""
    HOT = "hot"
    WARM = "warm"
    COOL = "cool"
    COLD = "cold"

@dataclass
class ScoringDetails:
    """Detailed scoring information for leads."""
    magnitude_score: float = 0.0
    location_score: float = 0.0
    property_value_score: float = 0.0
    seasonal_score: float = 0.0
    historical_score: float = 0.0
    confidence_score: float = 0.0
    final_score: float = 0.0
    tier: LeadTier = LeadTier.COLD
    components: Dict[str, float] = None
    
    def __post_init__(self):
        if self.components is None:
            self.components = {}

def load_config() -> APIConfig:
    """Load configuration from environment variables."""
    token = os.environ.get('NCEI_TOKEN')
    timeout = int(os.environ.get('NCEI_TIMEOUT', DEFAULT_API_TIMEOUT))
    rate_limit_delay = float(os.environ.get('NCEI_RATE_LIMIT_DELAY', DEFAULT_RATE_LIMIT_DELAY))
    max_retries = int(os.environ.get('NCEI_MAX_RETRIES', MAX_RETRIES))
    
    config = APIConfig(
        token=token,
        timeout=timeout,
        rate_limit_delay=rate_limit_delay,
        max_retries=max_retries
    )
    
    logger.info(f"Loaded configuration - Token: {'***' if token else 'None'}, "
                f"Timeout: {timeout}s, Rate limit: {rate_limit_delay}s, "
                f"Max retries: {max_retries}")
    
    return config

def iso_today_minus(days: int) -> str:
    """Get ISO date string for today minus specified days."""
    return (datetime.date.today() - datetime.timedelta(days=days)).isoformat()

def get_magnitude(rec: Dict[str, Any]) -> Optional[float]:
    """Extract magnitude from record with multiple field name support."""
    for key in ('MAGNITUDE', 'MAG', 'magnitude', 'mag'):
        val = rec.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None

def record_lat(rec: Dict[str, Any]) -> Optional[float]:
    """Extract latitude from record with multiple field name support."""
    for key in ('BEGIN_LAT', 'BEGIN_LATITUDE', 'begin_lat', 'BEGIN_LAT_DECIMAL'):
        v = rec.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    # fallback fields
    for key in ('LATITUDE', 'lat'):
        v = rec.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None

def record_lon(rec: Dict[str, Any]) -> Optional[float]:
    """Extract longitude from record with multiple field name support."""
    for key in ('BEGIN_LON', 'BEGIN_LONGITUDE', 'begin_lon', 'BEGIN_LON_DECIMAL'):
        v = rec.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    for key in ('LONGITUDE', 'lon'):
        v = rec.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None

def validate_event(event: Dict[str, Any]) -> bool:
    """Validate that an event has required fields."""
    required_fields = ['EVENT_TYPE', 'STATE']
    for field in required_fields:
        if not event.get(field):
            logger.warning(f"Event missing required field: {field}")
            return False
    
    lat = record_lat(event)
    lon = record_lon(event)
    
    if lat is None or lon is None:
        logger.warning("Event missing valid latitude/longitude")
        return False
    
    # Validate lat/lon ranges
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        logger.warning(f"Event has invalid coordinates: lat={lat}, lon={lon}")
        return False
    
    return True

def fetch_ncei_with_retry(config: APIConfig, start: str, end: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """Fetch NCEI data with retry logic and proper error handling."""
    params = {
        'dataset': config.dataset,
        'startDate': start,
        'endDate': end,
        'limit': limit,
        'format': 'json',
    }
    headers = {'Accept': 'application/json'}
    if config.token:
        headers['token'] = config.token

    results: List[Dict[str, Any]] = []
    offset = 1
    total_fetched = 0
    
    logger.info(f"Starting NCEI data fetch from {start} to {end}")
    
    for attempt in range(config.max_retries):
        try:
            while True:
                params['offset'] = offset
                logger.info(f"Fetching offset={offset}, attempt={attempt + 1}")
                
                # Rate limiting
                time.sleep(config.rate_limit_delay)
                
                resp = requests.get(
                    config.base_url, 
                    params=params, 
                    headers=headers, 
                    timeout=config.timeout
                )
                
                # Handle rate limiting
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get('Retry-After', config.rate_limit_delay))
                    logger.warning(f"Rate limited, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                resp.raise_for_status()
                
                data = resp.json()
                batch = data.get('results') or data.get('data') or data
                
                if isinstance(batch, dict):
                    batch = batch.get('results', [])
                
                if not batch:
                    logger.info("No more data available")
                    break
                
                # Validate and filter events
                valid_batch = []
                for event in batch:
                    if validate_event(event):
                        valid_batch.append(event)
                    else:
                        logger.debug(f"Skipping invalid event: {event.get('EVENT_ID', 'unknown')}")
                
                results.extend(valid_batch)
                total_fetched += len(valid_batch)
                
                logger.info(f"Fetched {len(valid_batch)} valid events (total: {total_fetched})")
                
                if len(batch) < limit:
                    break
                
                offset += limit
            
            logger.info(f"Successfully fetched {len(results)} valid events")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed (attempt {attempt + 1}/{config.max_retries}): {e}")
            if attempt < config.max_retries - 1:
                wait_time = (2 ** attempt) * config.rate_limit_delay
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries exceeded, giving up")
                raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    return results

def filter_events(events: List[Dict[str, Any]], hail_min: float, wind_min: float) -> List[Dict[str, Any]]:
    """Filter events based on criteria for Wisconsin and Illinois."""
    matches: List[Dict[str, Any]] = []
    
    logger.info(f"Filtering {len(events)} events with hail_min={hail_min}, wind_min={wind_min}")
    
    for rec in events:
        etype = (rec.get('EVENT_TYPE') or rec.get('eventType') or '').lower()
        mag = get_magnitude(rec)
        state = (rec.get('STATE') or rec.get('state') or '')
        lat = record_lat(rec)

        # Check if in Wisconsin
        is_wi = state and (state.upper() in ('WISCONSIN', 'WI') or state == '55' or state.upper().startswith('WI'))
        
        # Check if in Illinois (excluding southern IL below 41.5°N)
        is_il = state and (state.upper() in ('ILLINOIS', 'IL') or state == '17' or state.upper().startswith('IL'))
        if is_il and lat is not None and lat < 41.5:
            is_il = False

        # Filter hail events
        if 'hail' in etype:
            if mag is not None and mag >= hail_min and (is_wi or is_il):
                matches.append(rec)
                logger.debug(f"Matched hail event: {rec.get('EVENT_ID')} - magnitude: {mag}")
            continue
        
        # Filter wind events
        if 'wind' in etype:
            if mag is not None and mag >= wind_min and (is_wi or is_il):
                matches.append(rec)
                logger.debug(f"Matched wind event: {rec.get('EVENT_ID')} - magnitude: {mag}")
            continue
    
    logger.info(f"Filtered to {len(matches)} relevant events")
    return matches

def calculate_enhanced_lead_score(event: Dict[str, Any]) -> Tuple[float, Dict[str, float], float, LeadTier]:
    """Calculate enhanced lead score with multiple factors."""
    mag = get_magnitude(event) or 0.0
    lat = record_lat(event)
    lon = record_lon(event)
    event_type = (event.get('EVENT_TYPE') or '').lower()
    state = (event.get('STATE') or '').upper()
    
    # Initialize component scores
    components = {
        'magnitude_score': 0.0,
        'location_score': 0.0,
        'property_value_score': 0.0,
        'seasonal_score': 0.0,
        'historical_score': 0.0
    }
    
    # 1. Magnitude Score (35% weight)
    if 'hail' in event_type:
        # Exponential scaling for hail - larger hail is exponentially more damaging
        components['magnitude_score'] = min(35, (mag ** 1.5) * 15)
    elif 'wind' in event_type:
        # Logarithmic scaling for wind - diminishing returns for very high winds
        components['magnitude_score'] = min(35, math.log(mag + 1) * 12)
    elif 'tornado' in event_type:
        # Tornado magnitude (EF scale) gets high base score
        components['magnitude_score'] = min(35, mag * 10)
    
    # 2. Geographic Location Score (25% weight)
    if lat and lon:
        # High-value metropolitan areas in WI/IL
        metro_areas = {
            'chicago': (41.8781, -87.6298, 25),
            'milwaukee': (43.0389, -87.9065, 20),
            'madison': (43.0731, -89.4012, 18),
            'green_bay': (44.5133, -88.0133, 15),
            'rockford': (42.2711, -89.0940, 12)
        }
        
        max_location_score = 0
        for city, (city_lat, city_lon, base_score) in metro_areas.items():
            distance = calculate_distance(lat, lon, city_lat, city_lon)
            if distance <= 50:  # Within 50 miles
                # Score decreases with distance
                distance_score = base_score * (1 - distance / 50)
                max_location_score = max(max_location_score, distance_score)
        
        components['location_score'] = max_location_score
        
        # Additional score for being in target states
        if state in ['WISCONSIN', 'WI', 'ILLINOIS', 'IL']:
            components['location_score'] += 5
    
    # 3. Property Value Score (20% weight)
    if lat and lon:
        # Simulate property value based on location
        if state in ['ILLINOIS', 'IL']:
            if lat > 41.8:  # Northern IL (Chicago area)
                components['property_value_score'] = 18
            else:
                components['property_value_score'] = 12
        elif state in ['WISCONSIN', 'WI']:
            if lon < -88.5:  # Eastern WI (Milwaukee/Madison area)
                components['property_value_score'] = 15
            else:
                components['property_value_score'] = 10
    
    # 4. Seasonal Score (10% weight)
    event_date = event.get('BEGIN_DATE_TIME', '')
    if event_date:
        try:
            # Parse date and determine season
            if 'T' in event_date:
                date_part = event_date.split('T')[0]
            else:
                date_part = event_date
            
            event_datetime = datetime.datetime.fromisoformat(date_part.replace('Z', '+00:00'))
            month = event_datetime.month
            
            # Peak hail season is May-August
            if 5 <= month <= 8:
                components['seasonal_score'] = 10
            elif 4 <= month <= 9:
                components['seasonal_score'] = 7
            else:
                components['seasonal_score'] = 3
        except:
            components['seasonal_score'] = 5
    else:
        components['seasonal_score'] = 5
    
    # 5. Historical Patterns Score (10% weight)
    # Simulate historical storm frequency by region
    if state in ['ILLINOIS', 'IL']:
        components['historical_score'] = 8
    elif state in ['WISCONSIN', 'WI']:
        components['historical_score'] = 7
    else:
        components['historical_score'] = 3
    
    # Calculate final score (weighted sum)
    final_score = sum(components.values())
    
    # Calculate confidence score based on data quality
    confidence = 0.0
    if mag > 0:
        confidence += 30
    if lat and lon:
        confidence += 40
    if state:
        confidence += 20
    if event_type:
        confidence += 10
    
    # Determine lead tier
    if final_score >= 80:
        tier = LeadTier.HOT.value
    elif final_score >= 60:
        tier = LeadTier.WARM.value
    elif final_score >= 40:
        tier = LeadTier.COOL.value
    else:
        tier = LeadTier.COLD.value
    
    return final_score, components, confidence, tier

def make_lead_from_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Create a comprehensive lead record from a storm event."""
    ev_id = ev.get('EVENT_ID') or ev.get('eventId') or ev.get('id') or f"ev-{random.randint(100000, 999999)}"
    mag = get_magnitude(ev) or 0.0
    lat = record_lat(ev)
    lon = record_lon(ev)
    
    # Calculate enhanced lead score
    final_score, component_scores, confidence, tier = calculate_enhanced_lead_score(ev)
    
    # Create scoring details
    scoring_details = ScoringDetails(
        magnitude_score=component_scores['magnitude_score'],
        location_score=component_scores['location_score'],
        property_value_score=component_scores['property_value_score'],
        seasonal_score=component_scores['seasonal_score'],
        historical_score=component_scores['historical_score'],
        confidence_score=confidence,
        final_score=final_score,
        tier=tier,
        components=component_scores
    )
    
    lead = {
        'lead_id': f'lead-{ev_id}',
        'event_id': ev_id,
        'score': final_score,
        'status': 'new',
        'tier': tier.value,
        'confidence': confidence,
        'created_ts': datetime.datetime.utcnow().isoformat() + 'Z',
        'event': ev,
        'property': {
            'lat': lat,
            'lon': lon,
            'address': '',
            'property_id': '',
            'geocode_data': None,
            'enrichment_data': None
        },
        'contact': None,
        'provenance': {
            'source': 'ncei', 
            'ingested_ts': datetime.datetime.utcnow().isoformat() + 'Z',
            'event_type': ev.get('EVENT_TYPE'),
            'magnitude': mag,
            'state': ev.get('STATE'),
            'data_quality': confidence
        },
        'scoring_details': asdict(scoring_details),
        'enrichment_metadata': {
            'enriched': False,
            'enrichment_version': '1.0',
            'data_sources': [],
            'quality_score': 0
        }
    }
    return lead

def synthetic_leads(n: int = 10) -> List[Dict[str, Any]]:
    """Generate synthetic leads for demo purposes."""
    leads: List[Dict[str, Any]] = []
    
    logger.info(f"Generating {n} synthetic leads")
    
    for i in range(n):
        # Weight towards Wisconsin (60%) vs Illinois (40%)
        if random.random() < 0.6:
            lat = random.uniform(43.0, 45.0)  # Wisconsin
            state = 'WI'
        else:
            lat = random.uniform(41.6, 43.0)  # Illinois (northern)
            state = 'IL'
        
        lon = random.uniform(-93.0, -87.0)
        mag = round(random.uniform(0.5, 2.0), 2)
        
        ev = {
            'EVENT_ID': f'syn-{i}',
            'EVENT_TYPE': 'Hail',
            'MAGNITUDE': mag,
            'BEGIN_LAT': lat,
            'BEGIN_LON': lon,
            'STATE': state
        }
        leads.append(make_lead_from_event(ev))
    
    logger.info(f"Generated {len(leads)} synthetic leads")
    return leads

def write_leads(leads: List[Dict[str, Any]]) -> None:
    """Write leads to JSONL file."""
    out = DATA_DIR / 'leads.jsonl'
    
    try:
        with out.open('w', encoding='utf-8') as f:
            for lead in leads:
                f.write(json.dumps(lead, ensure_ascii=False) + '\n')
        logger.info(f'Successfully wrote {len(leads)} leads to {out}')
    except Exception as e:
        logger.error(f"Failed to write leads file: {e}")
        raise

def write_raw(events: List[Dict[str, Any]], start: str, end: str) -> None:
    """Write raw events to JSON file."""
    ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    raw_file = DATA_DIR / f'ncei_raw_{start}_{end}_{ts}.json'
    
    try:
        with raw_file.open('w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
        logger.info(f'Successfully wrote raw events to {raw_file}')
    except Exception as e:
        logger.error(f"Failed to write raw events file: {e}")
        raise

def load_cached_data(start: str, end: str) -> Optional[List[Dict[str, Any]]]:
    """Load cached data if available."""
    cache_dir = DATA_DIR / 'cache'
    cache_dir.mkdir(exist_ok=True)
    
    cache_file = cache_dir / f'ncei_cache_{start}_{end}.json'
    
    if cache_file.exists():
        try:
            with cache_file.open('r', encoding='utf-8') as f:
                cached_data = json.load(f)
            logger.info(f"Loaded cached data from {cache_file}")
            return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
    
    return None

def cache_data(events: List[Dict[str, Any]], start: str, end: str) -> None:
    """Cache data for future use."""
    cache_dir = DATA_DIR / 'cache'
    cache_dir.mkdir(exist_ok=True)
    
    cache_file = cache_dir / f'ncei_cache_{start}_{end}.json'
    
    try:
        with cache_file.open('w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
        logger.info(f"Cached data to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to cache data: {e}")

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in miles."""
    # Haversine formula
    R = 3959  # Earth's radius in miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def is_within_radius(lat1: float, lon1: float, lat2: float, lon2: float, radius_miles: float) -> bool:
    """Check if two coordinates are within a given radius."""
    distance = calculate_distance(lat1, lon1, lat2, lon2)
    return distance <= radius_miles

def create_storm_footprint(event: Dict[str, Any]) -> Dict[str, Any]:
    """Create a storm footprint polygon for geospatial processing."""
    lat = record_lat(event)
    lon = record_lon(event)
    mag = get_magnitude(event) or 0.0
    event_type = (event.get('EVENT_TYPE') or '').lower()
    
    if not lat or not lon:
        return {}
    
    # Calculate footprint radius based on event type and magnitude
    if 'hail' in event_type:
        # Hail footprint: approximately 1 mile per inch of hail
        radius_miles = max(1.0, mag * 1.5)
    elif 'wind' in event_type:
        # Wind footprint: approximately 2 miles per 20 mph of wind
        radius_miles = max(2.0, (mag / 20) * 2)
    elif 'tornado' in event_type:
        # Tornado footprint: based on EF scale
        radius_miles = max(0.5, mag * 2)
    else:
        radius_miles = 1.0
    
    # Create simple circular footprint (in real implementation, would use actual polygons)
    footprint = {
        'type': 'circle',
        'center': {'lat': lat, 'lon': lon},
        'radius_miles': radius_miles,
        'event_type': event_type,
        'magnitude': mag,
        'event_id': event.get('EVENT_ID', 'unknown')
    }
    
    return footprint

def find_properties_in_footprint(footprint: Dict[str, Any], enricher: AddressEnricher, 
                               config: PropertyEnrichmentConfig) -> List[Dict[str, Any]]:
    """Find properties within a storm footprint."""
    center_lat = footprint['center']['lat']
    center_lon = footprint['center']['lon']
    radius_miles = footprint['radius_miles']
    
    properties = []
    
    try:
        # Generate property candidates around the storm center
        # In real implementation, this would query OpenAddresses or other property databases
        num_properties = min(config.max_properties_per_event, int(radius_miles * 10))
        
        for i in range(num_properties):
            # Generate random properties within the radius
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius_miles)
            
            # Convert polar to cartesian coordinates
            lat_offset = distance * math.cos(angle) / 69.0  # 1 degree ≈ 69 miles
            lon_offset = distance * math.sin(angle) / (69.0 * math.cos(math.radians(center_lat)))
            
            prop_lat = center_lat + lat_offset
            prop_lon = center_lon + lon_offset
            
            # Enrich the property data
            property_data = enricher.enrich_property_data(prop_lat, prop_lon)
            
            # Add storm-specific information
            property_data['storm_distance'] = distance
            property_data['storm_bearing'] = math.degrees(angle)
            property_data['footprint_id'] = footprint['event_id']
            
            properties.append(property_data)
            
            # Rate limiting
            if i < num_properties - 1:
                time.sleep(config.rate_limit_delay)
    
    except Exception as e:
        logger.error(f"Error finding properties in footprint: {e}")
    
    return properties

def deduplicate_leads(leads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate leads based on property location and event proximity."""
    unique_leads = []
    seen_properties = set()
    
    for lead in leads:
        property_data = lead.get('property', {})
        lat = property_data.get('lat')
        lon = property_data.get('lon')
        
        if lat and lon:
            # Create a unique key based on coordinates and nearby events
            coord_key = f"{lat:.4f}_{lon:.4f}"
            
            # Check for nearby events (within 1 mile)
            is_duplicate = False
            for seen_key in seen_properties:
                seen_lat, seen_lon = map(float, seen_key.split('_'))
                if is_within_radius(lat, lon, seen_lat, seen_lon, 1.0):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_properties.add(coord_key)
                unique_leads.append(lead)
    
    logger.info(f"Deduplicated leads: {len(leads)} -> {len(unique_leads)}")
    return unique_leads

def enrich_leads_with_properties(leads: List[Dict[str, Any]], enricher: AddressEnricher,
                                config: PropertyEnrichmentConfig) -> List[Dict[str, Any]]:
    """Enrich leads with comprehensive property data."""
    if not config.enable_enrichment:
        logger.info("Property enrichment disabled, skipping...")
        return leads
    
    enriched_leads = []
    total_quality_score = 0
    
    logger.info(f"Enriching {len(leads)} leads with property data...")
    
    for i, lead in enumerate(leads):
        try:
            logger.info(f"Enriching lead {i+1}/{len(leads)}: {lead.get('lead_id')}")
            
            # Extract coordinates from lead
            property_data = lead.get('property', {})
            lat = property_data.get('lat')
            lon = property_data.get('lon')
            
            if lat and lon:
                # Create storm footprint
                footprint = create_storm_footprint(lead.get('event', {}))
                
                # Find properties in the storm footprint
                properties = find_properties_in_footprint(footprint, enricher, config)
                
                # Geocode the location
                geocode_result = enricher.geocode_address("", lat, lon)
                
                # Update lead with enriched data
                lead['property'].update({
                    'geocode_data': geocode_result,
                    'storm_footprint': footprint,
                    'properties_in_footprint': len(properties),
                    'property_id': enricher.generate_property_id(lat, lon, geocode_result.get('normalized_address', ''))
                })
                
                # Add best property match
                if properties:
                    best_property = max(properties, key=lambda p: p.get('quality_score', 0))
                    lead['property']['enrichment_data'] = best_property
                    total_quality_score += best_property.get('quality_score', 0)
                
                # Update enrichment metadata
                lead['enrichment_metadata'].update({
                    'enriched': True,
                    'enriched_at': time.time(),
                    'data_sources': ['nominatim', 'simulated_property_data'],
                    'quality_score': best_property.get('quality_score', 0) if properties else 0,
                    'properties_found': len(properties)
                })
                
                # Update provenance
                lead['provenance']['enrichment_sources'] = ['nominatim', 'simulated_property_data']
                lead['provenance']['enrichment_quality'] = best_property.get('quality_score', 0) if properties else 0
            
            enriched_leads.append(lead)
            
            # Rate limiting
            if i < len(leads) - 1:
                time.sleep(config.rate_limit_delay)
        
        except Exception as e:
            logger.error(f"Error enriching lead {lead.get('lead_id')}: {e}")
            lead['enrichment_error'] = str(e)
            enriched_leads.append(lead)
    
    # Calculate average quality score
    avg_quality = total_quality_score / len(enriched_leads) if enriched_leads else 0
    logger.info(f"Property enrichment completed. Average quality score: {avg_quality:.2f}")
    
    return enriched_leads

def ingest_data(config: APIConfig, start: str, end: str, hail_min: float, wind_min: float, limit: int,
                enrichment_config: PropertyEnrichmentConfig = None) -> IngestionResult:
    """Main data ingestion function with fallback mechanisms and property enrichment."""
    start_time = time.time()
    
    # Initialize enrichment config if not provided
    if enrichment_config is None:
        enrichment_config = PropertyEnrichmentConfig()
    
    # Initialize address enricher
    enricher = None
    if enrichment_config.enable_enrichment:
        try:
            enricher = create_enricher()
            logger.info("Address enricher initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize address enricher: {e}")
            enrichment_config.enable_enrichment = False
    
    try:
        # Try to fetch from NOAA API if token is available
        if config.token:
            logger.info("Attempting to fetch data from NOAA API")
            try:
                events = fetch_ncei_with_retry(config, start, end, limit)
                
                if events:
                    # Cache the successful fetch
                    cache_data(events, start, end)
                    
                    filtered = filter_events(events, hail_min, wind_min)
                    leads = [make_lead_from_event(ev) for ev in filtered]
                    
                    if leads:
                        # Apply property enrichment if enabled
                        if enrichment_config.enable_enrichment and enricher:
                            logger.info("Applying property enrichment...")
                            leads = enrich_leads_with_properties(leads, enricher, enrichment_config)
                            
                            # Apply deduplication
                            leads = deduplicate_leads(leads)
                        
                        write_raw(events, start, end)
                        write_leads(leads)
                        
                        processing_time = time.time() - start_time
                        enriched_count = sum(1 for lead in leads if lead.get('enrichment_metadata', {}).get('enriched', False))
                        avg_quality = sum(lead.get('enrichment_metadata', {}).get('quality_score', 0) for lead in leads) / len(leads) if leads else 0
                        
                        logger.info(f"Successfully ingested {len(leads)} leads from NOAA API in {processing_time:.2f}s")
                        logger.info(f"Enriched leads: {enriched_count}, Average quality: {avg_quality:.2f}")
                        
                        return IngestionResult(
                            source=DataSource.NOAA_API,
                            events_count=len(events),
                            leads_count=len(leads),
                            success=True,
                            processing_time=processing_time,
                            enriched_leads_count=enriched_count,
                            quality_score_avg=avg_quality
                        )
                    else:
                        logger.warning("No leads found from NOAA API data")
                
            except Exception as e:
                logger.error(f"NOAA API fetch failed: {e}")
        
        # Try to load cached data
        logger.info("Attempting to load cached data")
        cached_events = load_cached_data(start, end)
        if cached_events:
            filtered = filter_events(cached_events, hail_min, wind_min)
            leads = [make_lead_from_event(ev) for ev in filtered]
            
            if leads:
                # Apply property enrichment if enabled
                if enrichment_config.enable_enrichment and enricher:
                    logger.info("Applying property enrichment to cached data...")
                    leads = enrich_leads_with_properties(leads, enricher, enrichment_config)
                    
                    # Apply deduplication
                    leads = deduplicate_leads(leads)
                
                write_leads(leads)
                
                processing_time = time.time() - start_time
                enriched_count = sum(1 for lead in leads if lead.get('enrichment_metadata', {}).get('enriched', False))
                avg_quality = sum(lead.get('enrichment_metadata', {}).get('quality_score', 0) for lead in leads) / len(leads) if leads else 0
                
                logger.info(f"Successfully ingested {len(leads)} leads from cached data in {processing_time:.2f}s")
                logger.info(f"Enriched leads: {enriched_count}, Average quality: {avg_quality:.2f}")
                
                return IngestionResult(
                    source=DataSource.CACHED,
                    events_count=len(cached_events),
                    leads_count=len(leads),
                    success=True,
                    processing_time=processing_time,
                    enriched_leads_count=enriched_count,
                    quality_score_avg=avg_quality
                )
        
        # Fallback to synthetic data
        logger.info("Falling back to synthetic data generation")
        leads = synthetic_leads(12)
        
        # Apply property enrichment if enabled
        if enrichment_config.enable_enrichment and enricher:
            logger.info("Applying property enrichment to synthetic data...")
            leads = enrich_leads_with_properties(leads, enricher, enrichment_config)
            
            # Apply deduplication
            leads = deduplicate_leads(leads)
        
        write_leads(leads)
        
        processing_time = time.time() - start_time
        enriched_count = sum(1 for lead in leads if lead.get('enrichment_metadata', {}).get('enriched', False))
        avg_quality = sum(lead.get('enrichment_metadata', {}).get('quality_score', 0) for lead in leads) / len(leads) if leads else 0
        
        logger.info(f"Generated {len(leads)} synthetic leads in {processing_time:.2f}s")
        logger.info(f"Enriched leads: {enriched_count}, Average quality: {avg_quality:.2f}")
        
        return IngestionResult(
            source=DataSource.SYNTHETIC,
            events_count=0,
            leads_count=len(leads),
            success=True,
            processing_time=processing_time,
            enriched_leads_count=enriched_count,
            quality_score_avg=avg_quality
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Data ingestion failed after {processing_time:.2f}s: {e}")
        
        return IngestionResult(
            source=DataSource.SYNTHETIC,
            events_count=0,
            leads_count=0,
            success=False,
            error_message=str(e),
            processing_time=processing_time,
            enriched_leads_count=0,
            quality_score_avg=0.0
        )

def enrich_existing_leads(leads_file: str, enrichment_config: PropertyEnrichmentConfig = None) -> IngestionResult:
    """Enrich existing leads from a JSONL file."""
    start_time = time.time()
    
    if enrichment_config is None:
        enrichment_config = PropertyEnrichmentConfig()
    
    # Initialize address enricher
    if not enrichment_config.enable_enrichment:
        logger.info("Property enrichment disabled, skipping...")
        return IngestionResult(
            source=DataSource.CACHED,
            events_count=0,
            leads_count=0,
            success=True,
            processing_time=0.0,
            enriched_leads_count=0,
            quality_score_avg=0.0
        )
    
    try:
        # Load existing leads
        leads = []
        with open(leads_file, 'r', encoding='utf-8') as f:
            for line in f:
                lead = json.loads(line.strip())
                leads.append(lead)
        
        logger.info(f"Loaded {len(leads)} existing leads from {leads_file}")
        
        # Initialize enricher
        enricher = create_enricher()
        
        # Apply enrichment
        enriched_leads = enrich_leads_with_properties(leads, enricher, enrichment_config)
        
        # Apply deduplication
        enriched_leads = deduplicate_leads(enriched_leads)
        
        # Write enriched leads
        write_leads(enriched_leads)
        
        processing_time = time.time() - start_time
        enriched_count = sum(1 for lead in enriched_leads if lead.get('enrichment_metadata', {}).get('enriched', False))
        avg_quality = sum(lead.get('enrichment_metadata', {}).get('quality_score', 0) for lead in enriched_leads) / len(enriched_leads) if enriched_leads else 0
        
        logger.info(f"Successfully enriched {enriched_count} leads in {processing_time:.2f}s")
        logger.info(f"Average quality score: {avg_quality:.2f}")
        
        return IngestionResult(
            source=DataSource.CACHED,
            events_count=0,
            leads_count=len(enriched_leads),
            success=True,
            processing_time=processing_time,
            enriched_leads_count=enriched_count,
            quality_score_avg=avg_quality
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Lead enrichment failed after {processing_time:.2f}s: {e}")
        
        return IngestionResult(
            source=DataSource.CACHED,
            events_count=0,
            leads_count=0,
            success=False,
            error_message=str(e),
            processing_time=processing_time,
            enriched_leads_count=0,
            quality_score_avg=0.0
        )

def main(argv=None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced Hail Hero data ingestion with property enrichment')
    today = datetime.date.today()
    default_end = today.isoformat()
    default_start = (today - datetime.timedelta(days=365)).isoformat()
    
    parser.add_argument('--start', default=default_start, help='Start date (ISO format)')
    parser.add_argument('--end', default=default_end, help='End date (ISO format)')
    parser.add_argument('--hail-min', type=float, default=DEFAULT_HAIL_MIN, help='Minimum hail size (inches)')
    parser.add_argument('--wind-min', type=float, default=DEFAULT_WIND_MIN, help='Minimum wind speed (mph)')
    parser.add_argument('--limit', type=int, default=1000, help='Maximum records per API call')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--enrich', action='store_true', help='Enable address and property enrichment')
    parser.add_argument('--enrich-only', action='store_true', help='Only enrich existing leads (no new data fetch)')
    parser.add_argument('--leads-file', default=str(DATA_DIR / 'leads.jsonl'), help='Path to leads file for enrichment-only mode')
    parser.add_argument('--max-properties', type=int, default=50, help='Maximum properties to find per event')
    parser.add_argument('--search-radius', type=float, default=5.0, help='Search radius in miles for property lookup')
    parser.add_argument('--quality-threshold', type=float, default=60.0, help='Quality threshold for lead filtering')
    parser.add_argument('--no-cache', action='store_true', help='Disable property caching')
    parser.add_argument('--no-openaddresses', action='store_true', help='Disable OpenAddresses integration')
    parser.add_argument('--no-nominatim', action='store_true', help='Disable Nominatim integration')
    parser.add_argument('--rate-limit', type=float, default=1.0, help='Rate limit delay in seconds')
    
    args = parser.parse_args(argv)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting Hail Hero data ingestion from {args.start} to {args.end}")
    
    try:
        config = load_config()
        
        # Configure property enrichment
        enrichment_config = PropertyEnrichmentConfig(
            enable_enrichment=args.enrich or args.enrich_only,
            max_properties_per_event=args.max_properties,
            search_radius_miles=args.search_radius,
            property_cache_enabled=not args.no_cache,
            openaddresses_enabled=not args.no_openaddresses,
            nominatim_enabled=not args.no_nominatim,
            rate_limit_delay=args.rate_limit,
            quality_threshold=args.quality_threshold
        )
        
        if args.enrich_only:
            # Enrich existing leads only
            logger.info(f"Enriching existing leads from {args.leads_file}")
            result = enrich_existing_leads(args.leads_file, enrichment_config)
        else:
            # Full data ingestion with optional enrichment
            result = ingest_data(config, args.start, args.end, args.hail_min, args.wind_min, args.limit, enrichment_config)
        
        if result.success:
            logger.info(f"Successfully completed operation: {result}")
            if result.enriched_leads_count > 0:
                logger.info(f"Enriched {result.enriched_leads_count} leads with average quality score: {result.quality_score_avg:.2f}")
            return 0
        else:
            logger.error(f"Operation failed: {result.error_message}")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())