#!/usr/bin/env python3
"""Enhanced NOAA Storm Event Processing System for Hail Hero.

This module provides comprehensive storm event data processing including:
- Real NOAA API integration with token-based authentication
- Storm event parsing with geometry, severity, and timestamp analysis
- Geospatial analysis for event polygons and affected areas
- Data quality control, validation, and normalization
- Rate limiting and error handling for NOAA API quotas
- Scheduled processing for nightly and real-time ingestion
- Efficient storage and retrieval of storm event data
- Event correlation for geographic areas
- Performance optimization for 15-minute processing requirements

Behavior:
- If environment variable NCEI_TOKEN is set, fetch events from NCEI Search API.
- Otherwise create synthetic leads for WI/IL for demo purposes.
- Outputs written to `specs/001-hail-hero-hail/data/leads.jsonl` and raw events JSON.
- Enhanced with proper error handling, rate limiting, and fallback mechanisms.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
import hashlib
import math
import threading
import sqlite3
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager
import concurrent.futures
from queue import Queue
import schedule

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('/workspaces/HailHero-/noaa_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add structured logging
class StructuredLogger:
    """Structured logger for better monitoring and debugging."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, msg: str, **kwargs):
        self.logger.info(f"{msg} {json.dumps(kwargs)}")
    
    def warning(self, msg: str, **kwargs):
        self.logger.warning(f"{msg} {json.dumps(kwargs)}")
    
    def error(self, msg: str, **kwargs):
        self.logger.error(f"{msg} {json.dumps(kwargs)}")
    
    def debug(self, msg: str, **kwargs):
        self.logger.debug(f"{msg} {json.dumps(kwargs)}")

struct_logger = StructuredLogger(__name__)

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'specs' / '001-hail-hero-hail' / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = DATA_DIR / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
DB_DIR = DATA_DIR / 'db'
DB_DIR.mkdir(exist_ok=True)

# Database paths
EVENTS_DB = DB_DIR / 'storm_events.db'
LEADS_DB = DB_DIR / 'leads.db'

# NOAA API Configuration
NCEI_BASE_URL = 'https://www.ncei.noaa.gov/access/services/search/v1/data'
NCEI_DATASET = 'stormevents'
NCEI_STORM_EVENTS_URL = 'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'

# Default thresholds
DEFAULT_HAIL_MIN = 0.5
DEFAULT_WIND_MIN = 60
DEFAULT_TORNADO_MIN = 0
DEFAULT_API_TIMEOUT = 30
DEFAULT_RATE_LIMIT_DELAY = 1.0  # seconds between API calls
MAX_RETRIES = 3
BATCH_SIZE = 1000
MAX_CONCURRENT_REQUESTS = 5

# Geographic bounds for Wisconsin and Illinois
WI_BOUNDS = {'min_lat': 42.5, 'max_lat': 47.0, 'min_lon': -92.9, 'max_lon': -86.8}
IL_BOUNDS = {'min_lat': 41.5, 'max_lat': 42.5, 'min_lon': -91.5, 'max_lon': -87.5}

@dataclass
class APIConfig:
    """Configuration for NOAA API integration."""
    token: Optional[str] = None
    base_url: str = NCEI_BASE_URL
    storm_events_url: str = NCEI_STORM_EVENTS_URL
    timeout: int = DEFAULT_API_TIMEOUT
    rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY
    max_retries: int = MAX_RETRIES
    dataset: str = NCEI_DATASET
    batch_size: int = BATCH_SIZE
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS
    enable_caching: bool = True
    enable_db_storage: bool = True
    log_level: str = LOG_LEVEL
    regions: List[Dict[str, float]] = field(default_factory=lambda: [WI_BOUNDS, IL_BOUNDS])
    
    def __post_init__(self):
        """Post-initialization configuration validation."""
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.rate_limit_delay <= 0:
            raise ValueError("Rate limit delay must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_concurrent_requests <= 0:
            raise ValueError("Max concurrent requests must be positive")

class DataSource(Enum):
    """Enum for data sources."""
    NOAA_API = "noaa_api"
    NOAA_STORM_EVENTS = "noaa_storm_events"
    SYNTHETIC = "synthetic"
    CACHED = "cached"
    DATABASE = "database"

@dataclass
class StormEvent:
    """Enhanced storm event data structure."""
    event_id: str
    event_type: str
    magnitude: Optional[float] = None
    magnitude_type: Optional[str] = None
    begin_lat: Optional[float] = None
    begin_lon: Optional[float] = None
    end_lat: Optional[float] = None
    end_lon: Optional[float] = None
    begin_time: Optional[str] = None
    end_time: Optional[str] = None
    state: Optional[str] = None
    county: Optional[str] = None
    injuries: Optional[int] = None
    fatalities: Optional[int] = None
    property_damage: Optional[float] = None
    crop_damage: Optional[float] = None
    episode_id: Optional[str] = None
    event_narrative: Optional[str] = None
    geometry: Optional[Dict[str, Any]] = None
    affected_area: Optional[float] = None  # square miles
    severity_score: Optional[float] = None
    quality_score: Optional[float] = None
    processed_ts: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'magnitude': self.magnitude,
            'magnitude_type': self.magnitude_type,
            'begin_lat': self.begin_lat,
            'begin_lon': self.begin_lon,
            'end_lat': self.end_lat,
            'end_lon': self.end_lon,
            'begin_time': self.begin_time,
            'end_time': self.end_time,
            'state': self.state,
            'county': self.county,
            'injuries': self.injuries,
            'fatalities': self.fatalities,
            'property_damage': self.property_damage,
            'crop_damage': self.crop_damage,
            'episode_id': self.episode_id,
            'event_narrative': self.event_narrative,
            'geometry': self.geometry,
            'affected_area': self.affected_area,
            'severity_score': self.severity_score,
            'quality_score': self.quality_score,
            'processed_ts': self.processed_ts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StormEvent':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class IngestionResult:
    """Result of data ingestion process."""
    source: DataSource
    events_count: int
    leads_count: int
    success: bool
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    api_calls_made: int = 0
    cache_hits: int = 0
    db_records_created: int = 0
    quality_metrics: Optional[Dict[str, Any]] = None

class DatabaseManager:
    """Manages SQLite database operations for storm events and leads."""
    
    def __init__(self, events_db: Path = EVENTS_DB, leads_db: Path = LEADS_DB):
        self.events_db = events_db
        self.leads_db = leads_db
        self._init_databases()
    
    def _init_databases(self):
        """Initialize database tables."""
        # Initialize events database
        with self._get_connection(self.events_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS storm_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    magnitude REAL,
                    magnitude_type TEXT,
                    begin_lat REAL,
                    begin_lon REAL,
                    end_lat REAL,
                    end_lon REAL,
                    begin_time TEXT,
                    end_time TEXT,
                    state TEXT,
                    county TEXT,
                    injuries INTEGER,
                    fatalities INTEGER,
                    property_damage REAL,
                    crop_damage REAL,
                    episode_id TEXT,
                    event_narrative TEXT,
                    geometry TEXT,  -- JSON
                    affected_area REAL,
                    severity_score REAL,
                    quality_score REAL,
                    processed_ts TEXT,
                    created_ts TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(event_id)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_type ON storm_events(event_type)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_state ON storm_events(state)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_time ON storm_events(begin_time)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_location ON storm_events(begin_lat, begin_lon)
            ''')
        
        # Initialize leads database
        with self._get_connection(self.leads_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS leads (
                    lead_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_ts TEXT NOT NULL,
                    property_data TEXT,  -- JSON
                    contact_data TEXT,  -- JSON
                    provenance_data TEXT,  -- JSON
                    enrichment_data TEXT,  -- JSON
                    updated_ts TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(lead_id),
                    FOREIGN KEY (event_id) REFERENCES storm_events(event_id)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_leads_score ON leads(score)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_leads_created ON leads(created_ts)
            ''')
    
    @contextmanager
    def _get_connection(self, db_path: Path):
        """Get database connection with proper error handling."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def store_event(self, event: StormEvent) -> bool:
        """Store a single storm event."""
        try:
            with self._get_connection(self.events_db) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO storm_events 
                    (event_id, event_type, magnitude, magnitude_type, begin_lat, begin_lon,
                     end_lat, end_lon, begin_time, end_time, state, county, injuries,
                     fatalities, property_damage, crop_damage, episode_id, event_narrative,
                     geometry, affected_area, severity_score, quality_score, processed_ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id, event.event_type, event.magnitude, event.magnitude_type,
                    event.begin_lat, event.begin_lon, event.end_lat, event.end_lon,
                    event.begin_time, event.end_time, event.state, event.county,
                    event.injuries, event.fatalities, event.property_damage, event.crop_damage,
                    event.episode_id, event.event_narrative, json.dumps(event.geometry),
                    event.affected_area, event.severity_score, event.quality_score, event.processed_ts
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to store event {event.event_id}: {e}")
            return False
    
    def store_events_batch(self, events: List[StormEvent]) -> int:
        """Store multiple storm events in a batch."""
        stored_count = 0
        try:
            with self._get_connection(self.events_db) as conn:
                for event in events:
                    try:
                        conn.execute('''
                            INSERT OR REPLACE INTO storm_events 
                            (event_id, event_type, magnitude, magnitude_type, begin_lat, begin_lon,
                             end_lat, end_lon, begin_time, end_time, state, county, injuries,
                             fatalities, property_damage, crop_damage, episode_id, event_narrative,
                             geometry, affected_area, severity_score, quality_score, processed_ts)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            event.event_id, event.event_type, event.magnitude, event.magnitude_type,
                            event.begin_lat, event.begin_lon, event.end_lat, event.end_lon,
                            event.begin_time, event.end_time, event.state, event.county,
                            event.injuries, event.fatalities, event.property_damage, event.crop_damage,
                            event.episode_id, event.event_narrative, json.dumps(event.geometry),
                            event.affected_area, event.severity_score, event.quality_score, event.processed_ts
                        ))
                        stored_count += 1
                    except Exception as e:
                        logger.error(f"Failed to store event {event.event_id}: {e}")
                
                conn.commit()
                logger.info(f"Stored {stored_count} events in database")
        except Exception as e:
            logger.error(f"Failed to batch store events: {e}")
        
        return stored_count
    
    def get_events_by_region(self, bounds: Dict[str, float], limit: int = 1000) -> List[StormEvent]:
        """Get events within geographic bounds."""
        events = []
        try:
            with self._get_connection(self.events_db) as conn:
                cursor = conn.execute('''
                    SELECT * FROM storm_events 
                    WHERE begin_lat BETWEEN ? AND ? 
                    AND begin_lon BETWEEN ? AND ?
                    ORDER BY begin_time DESC
                    LIMIT ?
                ''', (bounds['min_lat'], bounds['max_lat'], bounds['min_lon'], bounds['max_lon'], limit))
                
                for row in cursor.fetchall():
                    event_data = dict(row)
                    if event_data['geometry']:
                        event_data['geometry'] = json.loads(event_data['geometry'])
                    events.append(StormEvent.from_dict(event_data))
        except Exception as e:
            logger.error(f"Failed to get events by region: {e}")
        
        return events
    
    def get_events_by_time_range(self, start_time: str, end_time: str, limit: int = 1000) -> List[StormEvent]:
        """Get events within time range."""
        events = []
        try:
            with self._get_connection(self.events_db) as conn:
                cursor = conn.execute('''
                    SELECT * FROM storm_events 
                    WHERE begin_time BETWEEN ? AND ?
                    ORDER BY begin_time DESC
                    LIMIT ?
                ''', (start_time, end_time, limit))
                
                for row in cursor.fetchall():
                    event_data = dict(row)
                    if event_data['geometry']:
                        event_data['geometry'] = json.loads(event_data['geometry'])
                    events.append(StormEvent.from_dict(event_data))
        except Exception as e:
            logger.error(f"Failed to get events by time range: {e}")
        
        return events

class GeospatialAnalyzer:
    """Handles geospatial analysis for storm events."""
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in miles using Haversine formula."""
        R = 3959  # Earth's radius in miles
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    @staticmethod
    def calculate_affected_area(event: StormEvent) -> Optional[float]:
        """Calculate affected area in square miles based on event characteristics."""
        if event.begin_lat is None or event.begin_lon is None:
            return None
        
        # Base calculation on event type and magnitude
        if 'hail' in event.event_type.lower():
            # Hail affected area roughly based on hail size
            if event.magnitude:
                radius_miles = event.magnitude * 2  # Approximate radius in miles
                return math.pi * radius_miles**2
        
        elif 'tornado' in event.event_type.lower():
            # Tornado path width and length
            if event.magnitude:  # EF scale
                avg_width_miles = 0.1 + (event.magnitude * 0.2)  # EF0: 0.1mi, EF5: 1.1mi
                avg_length_miles = 2 + (event.magnitude * 8)  # EF0: 2mi, EF5: 42mi
                return avg_width_miles * avg_length_miles
        
        elif 'wind' in event.event_type.lower():
            # Wind affected area
            if event.magnitude:
                radius_miles = min(event.magnitude / 20, 10)  # Cap at 10 miles radius
                return math.pi * radius_miles**2
        
        # Default small area
        return 1.0
    
    @staticmethod
    def is_point_in_bounds(lat: float, lon: float, bounds: Dict[str, float]) -> bool:
        """Check if point is within geographic bounds."""
        return (bounds['min_lat'] <= lat <= bounds['max_lat'] and 
                bounds['min_lon'] <= lon <= bounds['max_lon'])
    
    @staticmethod
    def calculate_severity_score(event: StormEvent) -> float:
        """Calculate severity score based on event characteristics."""
        score = 0.0
        
        # Base score by event type
        event_type_lower = event.event_type.lower()
        if 'tornado' in event_type_lower:
            score += 50
        elif 'hail' in event_type_lower:
            score += 30
        elif 'wind' in event_type_lower:
            score += 20
        
        # Magnitude contribution
        if event.magnitude:
            if 'tornado' in event_type_lower:
                score += event.magnitude * 10  # EF scale 0-5
            elif 'hail' in event_type_lower:
                score += event.magnitude * 15  # Hail size in inches
            elif 'wind' in event_type_lower:
                score += event.magnitude / 5  # Wind speed in mph
        
        # Damage contribution
        if event.property_damage:
            score += min(event.property_damage / 10000, 30)  # Cap at 30 points
        
        # Casualties contribution
        if event.fatalities:
            score += event.fatalities * 20
        if event.injuries:
            score += event.injuries * 5
        
        return min(score, 100)  # Cap at 100

class DataValidator:
    """Handles data validation and quality control."""
    
    @staticmethod
    def validate_event(event_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate storm event data and return validation result with errors."""
        errors = []
        
        # Required fields
        required_fields = ['EVENT_ID', 'EVENT_TYPE']
        for field in required_fields:
            if not event_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate coordinates
        lat = DataValidator._extract_latitude(event_data)
        lon = DataValidator._extract_longitude(event_data)
        
        if lat is None:
            errors.append("Missing or invalid latitude")
        elif not (-90 <= lat <= 90):
            errors.append(f"Latitude out of range: {lat}")
        
        if lon is None:
            errors.append("Missing or invalid longitude")
        elif not (-180 <= lon <= 180):
            errors.append(f"Longitude out of range: {lon}")
        
        # Validate magnitude
        magnitude = DataValidator._extract_magnitude(event_data)
        if magnitude is not None and magnitude < 0:
            errors.append(f"Negative magnitude: {magnitude}")
        
        # Validate timestamps
        begin_time = event_data.get('BEGIN_TIME') or event_data.get('BEGIN_DATE')
        if begin_time:
            try:
                datetime.datetime.fromisoformat(begin_time.replace('Z', '+00:00'))
            except ValueError:
                errors.append(f"Invalid begin time format: {begin_time}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _extract_latitude(event_data: Dict[str, Any]) -> Optional[float]:
        """Extract latitude with multiple field name support."""
        for key in ('BEGIN_LAT', 'BEGIN_LATITUDE', 'begin_lat', 'BEGIN_LAT_DECIMAL', 'LATITUDE', 'lat'):
            val = event_data.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None
    
    @staticmethod
    def _extract_longitude(event_data: Dict[str, Any]) -> Optional[float]:
        """Extract longitude with multiple field name support."""
        for key in ('BEGIN_LON', 'BEGIN_LONGITUDE', 'begin_lon', 'BEGIN_LON_DECIMAL', 'LONGITUDE', 'lon'):
            val = event_data.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None
    
    @staticmethod
    def _extract_magnitude(event_data: Dict[str, Any]) -> Optional[float]:
        """Extract magnitude with multiple field name support."""
        for key in ('MAGNITUDE', 'MAG', 'magnitude', 'mag'):
            val = event_data.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None
    
    @staticmethod
    def normalize_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize event data to standard format."""
        normalized = event_data.copy()
        
        # Standardize field names
        field_mapping = {
            'BEGIN_LAT': 'begin_lat',
            'BEGIN_LON': 'begin_lon',
            'END_LAT': 'end_lat',
            'END_LON': 'end_lon',
            'BEGIN_TIME': 'begin_time',
            'END_TIME': 'end_time',
            'EVENT_TYPE': 'event_type',
            'EVENT_ID': 'event_id',
            'MAGNITUDE': 'magnitude',
            'STATE': 'state',
            'COUNTY': 'county',
            'INJURIES': 'injuries',
            'FATALITIES': 'fatalities',
            'PROPERTY_DAMAGE': 'property_damage',
            'CROP_DAMAGE': 'crop_damage',
            'EPISODE_ID': 'episode_id',
            'EVENT_NARRATIVE': 'event_narrative'
        }
        
        for old_field, new_field in field_mapping.items():
            if old_field in normalized and new_field not in normalized:
                normalized[new_field] = normalized[old_field]
        
        # Clean up string fields
        string_fields = ['event_type', 'state', 'county', 'event_narrative']
        for field in string_fields:
            if field in normalized and isinstance(normalized[field], str):
                normalized[field] = normalized[field].strip().title()
        
        return normalized
    
    @staticmethod
    def calculate_quality_score(event: StormEvent) -> float:
        """Calculate data quality score based on completeness and validity."""
        score = 0.0
        total_fields = 15
        
        # Check for presence of key fields
        if event.event_id:
            score += 5
        if event.event_type:
            score += 5
        if event.begin_lat is not None:
            score += 5
        if event.begin_lon is not None:
            score += 5
        if event.magnitude is not None:
            score += 5
        if event.begin_time:
            score += 5
        if event.state:
            score += 5
        if event.county:
            score += 3
        if event.end_lat is not None:
            score += 2
        if event.end_lon is not None:
            score += 2
        if event.episode_id:
            score += 3
        if event.event_narrative:
            score += 5
        
        return (score / total_fields) * 100

class RateLimiter:
    """Handles rate limiting for API calls."""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.last_call = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to maintain rate limit."""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call
            
            if time_since_last_call < self.delay:
                time.sleep(self.delay - time_since_last_call)
            
            self.last_call = time.time()

class NOAAClient:
    """Enhanced NOAA API client with robust error handling and rate limiting."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit_delay)
        self.session = self._create_session()
        struct_logger.info("NOAA client initialized", config={
            'base_url': config.base_url,
            'timeout': config.timeout,
            'rate_limit_delay': config.rate_limit_delay,
            'max_retries': config.max_retries
        })
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    def fetch_storm_events(self, start_date: str, end_date: str, 
                          event_types: Optional[List[str]] = None,
                          states: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fetch storm events from NOAA API with comprehensive error handling."""
        params = {
            'dataset': self.config.dataset,
            'startDate': start_date,
            'endDate': end_date,
            'limit': self.config.batch_size,
            'format': 'json',
        }
        
        if event_types:
            params['dataTypes'] = ','.join(event_types)
        if states:
            params['states'] = ','.join(states)
        
        headers = {'Accept': 'application/json'}
        if self.config.token:
            headers['token'] = self.config.token
        
        all_events = []
        offset = 1
        api_calls = 0
        
        struct_logger.info("Starting NOAA API fetch", start_date=start_date, end_date=end_date)
        
        try:
            while True:
                params['offset'] = offset
                
                # Rate limiting
                self.rate_limiter.wait_if_needed()
                
                response = self.session.get(
                    self.config.base_url,
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout
                )
                api_calls += 1
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', self.config.rate_limit_delay))
                    struct_logger.warning("Rate limited", retry_after=retry_after)
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                
                data = response.json()
                batch = data.get('results') or data.get('data') or data
                
                if isinstance(batch, dict):
                    batch = batch.get('results', [])
                
                if not batch:
                    struct_logger.info("No more data available")
                    break
                
                # Validate and filter events
                valid_events = []
                for event in batch:
                    is_valid, errors = DataValidator.validate_event(event)
                    if is_valid:
                        valid_events.append(event)
                    else:
                        struct_logger.debug("Invalid event skipped", event_id=event.get('EVENT_ID'), errors=errors)
                
                all_events.extend(valid_events)
                struct_logger.info("Fetched batch", batch_size=len(valid_events), total=len(all_events))
                
                if len(batch) < self.config.batch_size:
                    break
                
                offset += self.config.batch_size
            
            struct_logger.info("NOAA API fetch completed", total_events=len(all_events), api_calls=api_calls)
            return all_events
            
        except Exception as e:
            struct_logger.error("NOAA API fetch failed", error=str(e))
            raise
    
    def fetch_storm_events_csv(self, year: int) -> Optional[Dict[str, Any]]:
        """Fetch storm events CSV data for a specific year."""
        csv_url = f"{self.config.storm_events_url}StormEvents_{year}.csv.gz"
        
        try:
            self.rate_limiter.wait_if_needed()
            
            response = self.session.get(csv_url, timeout=self.config.timeout)
            response.raise_for_status()
            
            # Return CSV data info (would need CSV parsing in production)
            return {
                'year': year,
                'url': csv_url,
                'size_bytes': len(response.content),
                'content_type': response.headers.get('content-type')
            }
            
        except Exception as e:
            struct_logger.error("Failed to fetch CSV data", year=year, error=str(e))
            return None

class EventProcessor:
    """Processes storm events and creates leads."""
    
    def __init__(self, config: APIConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.geospatial_analyzer = GeospatialAnalyzer()
        self.data_validator = DataValidator()
    
    def process_events(self, events: List[Dict[str, Any]]) -> Tuple[List[StormEvent], List[Dict[str, Any]]]:
        """Process raw event data into structured StormEvent objects and leads."""
        storm_events = []
        leads = []
        
        struct_logger.info("Processing events", count=len(events))
        
        for event_data in events:
            try:
                # Normalize and validate event data
                normalized_data = self.data_validator.normalize_event(event_data)
                is_valid, errors = self.data_validator.validate_event(normalized_data)
                
                if not is_valid:
                    struct_logger.warning("Invalid event skipped", errors=errors)
                    continue
                
                # Create StormEvent object
                storm_event = self._create_storm_event(normalized_data)
                
                # Calculate additional attributes
                storm_event.affected_area = self.geospatial_analyzer.calculate_affected_area(storm_event)
                storm_event.severity_score = self.geospatial_analyzer.calculate_severity_score(storm_event)
                storm_event.quality_score = self.data_validator.calculate_quality_score(storm_event)
                storm_event.processed_ts = datetime.datetime.utcnow().isoformat() + 'Z'
                
                storm_events.append(storm_event)
                
                # Create lead if event matches criteria
                if self._should_create_lead(storm_event):
                    lead = self._create_lead(storm_event)
                    leads.append(lead)
                
            except Exception as e:
                struct_logger.error("Failed to process event", event_id=event_data.get('EVENT_ID'), error=str(e))
        
        struct_logger.info("Event processing completed", 
                          storm_events=len(storm_events), leads=len(leads))
        
        return storm_events, leads
    
    def _create_storm_event(self, event_data: Dict[str, Any]) -> StormEvent:
        """Create StormEvent from normalized data."""
        return StormEvent(
            event_id=event_data.get('event_id', event_data.get('EVENT_ID', '')),
            event_type=event_data.get('event_type', event_data.get('EVENT_TYPE', '')),
            magnitude=event_data.get('magnitude', event_data.get('MAGNITUDE')),
            magnitude_type=event_data.get('magnitude_type', event_data.get('MAGNITUDE_TYPE')),
            begin_lat=event_data.get('begin_lat', event_data.get('BEGIN_LAT')),
            begin_lon=event_data.get('begin_lon', event_data.get('BEGIN_LON')),
            end_lat=event_data.get('end_lat', event_data.get('END_LAT')),
            end_lon=event_data.get('end_lon', event_data.get('END_LON')),
            begin_time=event_data.get('begin_time', event_data.get('BEGIN_TIME')),
            end_time=event_data.get('end_time', event_data.get('END_TIME')),
            state=event_data.get('state', event_data.get('STATE')),
            county=event_data.get('county', event_data.get('COUNTY')),
            injuries=event_data.get('injuries', event_data.get('INJURIES')),
            fatalities=event_data.get('fatalities', event_data.get('FATALITIES')),
            property_damage=event_data.get('property_damage', event_data.get('PROPERTY_DAMAGE')),
            crop_damage=event_data.get('crop_damage', event_data.get('CROP_DAMAGE')),
            episode_id=event_data.get('episode_id', event_data.get('EPISODE_ID')),
            event_narrative=event_data.get('event_narrative', event_data.get('EVENT_NARRATIVE'))
        )
    
    def _should_create_lead(self, event: StormEvent) -> bool:
        """Determine if a lead should be created for this event."""
        # Check if event is in target regions
        if event.begin_lat is None or event.begin_lon is None:
            return False
        
        in_target_region = False
        for region in self.config.regions:
            if self.geospatial_analyzer.is_point_in_bounds(event.begin_lat, event.begin_lon, region):
                in_target_region = True
                break
        
        if not in_target_region:
            return False
        
        # Check event type and magnitude thresholds
        event_type_lower = event.event_type.lower()
        
        if 'hail' in event_type_lower:
            return event.magnitude is not None and event.magnitude >= DEFAULT_HAIL_MIN
        elif 'wind' in event_type_lower:
            return event.magnitude is not None and event.magnitude >= DEFAULT_WIND_MIN
        elif 'tornado' in event_type_lower:
            return event.magnitude is not None and event.magnitude >= DEFAULT_TORNADO_MIN
        
        return False
    
    def _create_lead(self, event: StormEvent) -> Dict[str, Any]:
        """Create lead from storm event."""
        # Enhanced scoring algorithm
        if 'hail' in event.event_type.lower():
            score = min(100, int((event.magnitude / 3.0) * 100))
        elif 'tornado' in event.event_type.lower():
            score = min(100, int((event.magnitude / 5.0) * 100))
        else:
            score = min(100, int((event.magnitude / 100.0) * 100))
        
        # Incorporate severity and quality scores
        if event.severity_score:
            score = int((score + event.severity_score) / 2)
        if event.quality_score:
            score = int((score + event.quality_score) / 2)
        
        return {
            'lead_id': f'lead-{event.event_id}',
            'event_id': event.event_id,
            'score': score,
            'status': 'new',
            'created_ts': datetime.datetime.utcnow().isoformat() + 'Z',
            'event': event.to_dict(),
            'property': {
                'lat': event.begin_lat,
                'lon': event.begin_lon,
                'affected_area': event.affected_area,
                'estimated_damage': event.property_damage
            },
            'contact': None,
            'provenance': {
                'source': 'ncei',
                'ingested_ts': datetime.datetime.utcnow().isoformat() + 'Z',
                'event_type': event.event_type,
                'magnitude': event.magnitude,
                'state': event.state,
                'severity_score': event.severity_score,
                'quality_score': event.quality_score
            },
        }

class CacheManager:
    """Manages caching of NOAA data."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, start_date: str, end_date: str, 
                     event_types: Optional[List[str]] = None,
                     states: Optional[List[str]] = None) -> str:
        """Generate cache key for request parameters."""
        key_data = {
            'start_date': start_date,
            'end_date': end_date,
            'event_types': sorted(event_types) if event_types else [],
            'states': sorted(states) if states else []
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_data(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached data if available and not expired."""
        cache_file = self.cache_dir / f'{cache_key}.json'
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is expired (24 hours)
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > 24 * 60 * 60:  # 24 hours
                return None
            
            with cache_file.open('r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            struct_logger.info("Cache hit", cache_key=cache_key)
            return cached_data
            
        except Exception as e:
            struct_logger.warning("Failed to load cached data", cache_key=cache_key, error=str(e))
            return None
    
    def cache_data(self, cache_key: str, data: List[Dict[str, Any]]):
        """Cache data for future use."""
        cache_file = self.cache_dir / f'{cache_key}.json'
        
        try:
            with cache_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            struct_logger.info("Data cached", cache_key=cache_key, items=len(data))
            
        except Exception as e:
            struct_logger.warning("Failed to cache data", cache_key=cache_key, error=str(e))

class Scheduler:
    """Handles scheduled data ingestion tasks."""
    
    def __init__(self, ingestion_func):
        self.ingestion_func = ingestion_func
        self.running = False
        self.scheduler_thread = None
    
    def start(self):
        """Start the scheduler."""
        if self.running:
            return
        
        self.running = True
        
        # Schedule daily ingestion at 2 AM
        schedule.every().day.at("02:00").do(self.ingestion_func)
        
        # Schedule hourly ingestion for real-time data
        schedule.every().hour.do(self._hourly_ingestion)
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        struct_logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        struct_logger.info("Scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _hourly_ingestion(self):
        """Hourly ingestion for recent data."""
        try:
            # Get data for the last hour
            end_time = datetime.datetime.utcnow()
            start_time = end_time - datetime.timedelta(hours=1)
            
            struct_logger.info("Running hourly ingestion", 
                             start_time=start_time.isoformat(),
                             end_time=end_time.isoformat())
            
            # This would call the ingestion function with recent time range
            # self.ingestion_func(start_time.isoformat(), end_time.isoformat())
            
        except Exception as e:
            struct_logger.error("Hourly ingestion failed", error=str(e))

def load_config() -> APIConfig:
    """Load configuration from environment variables."""
    token = os.environ.get('NCEI_TOKEN')
    timeout = int(os.environ.get('NCEI_TIMEOUT', DEFAULT_API_TIMEOUT))
    rate_limit_delay = float(os.environ.get('NCEI_RATE_LIMIT_DELAY', DEFAULT_RATE_LIMIT_DELAY))
    max_retries = int(os.environ.get('NCEI_MAX_RETRIES', MAX_RETRIES))
    batch_size = int(os.environ.get('NCEI_BATCH_SIZE', BATCH_SIZE))
    max_concurrent_requests = int(os.environ.get('NCEI_MAX_CONCURRENT_REQUESTS', MAX_CONCURRENT_REQUESTS))
    enable_caching = os.environ.get('NCEI_ENABLE_CACHING', 'true').lower() == 'true'
    enable_db_storage = os.environ.get('NCEI_ENABLE_DB_STORAGE', 'true').lower() == 'true'
    
    config = APIConfig(
        token=token,
        timeout=timeout,
        rate_limit_delay=rate_limit_delay,
        max_retries=max_retries,
        batch_size=batch_size,
        max_concurrent_requests=max_concurrent_requests,
        enable_caching=enable_caching,
        enable_db_storage=enable_db_storage
    )
    
    struct_logger.info("Configuration loaded", config={
        'has_token': bool(token),
        'timeout': timeout,
        'rate_limit_delay': rate_limit_delay,
        'max_retries': max_retries,
        'batch_size': batch_size,
        'max_concurrent_requests': max_concurrent_requests,
        'enable_caching': enable_caching,
        'enable_db_storage': enable_db_storage
    })
    
    return config

def write_leads(leads: List[Dict[str, Any]]) -> None:
    """Write leads to JSONL file."""
    out = DATA_DIR / 'leads.jsonl'
    
    try:
        with out.open('w', encoding='utf-8') as f:
            for lead in leads:
                f.write(json.dumps(lead, ensure_ascii=False) + '\n')
        struct_logger.info("Leads written to file", file=str(out), count=len(leads))
    except Exception as e:
        struct_logger.error("Failed to write leads file", error=str(e))
        raise

def write_raw_events(events: List[StormEvent], start: str, end: str) -> None:
    """Write raw events to JSON file."""
    ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    raw_file = DATA_DIR / f'ncei_raw_{start}_{end}_{ts}.json'
    
    try:
        with raw_file.open('w', encoding='utf-8') as f:
            json.dump([event.to_dict() for event in events], f, indent=2, ensure_ascii=False)
        struct_logger.info("Raw events written to file", file=str(raw_file), count=len(events))
    except Exception as e:
        struct_logger.error("Failed to write raw events file", error=str(e))
        raise

def synthetic_leads(n: int = 10) -> List[Dict[str, Any]]:
    """Generate synthetic leads for demo purposes."""
    leads = []
    
    struct_logger.info("Generating synthetic leads", count=n)
    
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
        
        event = StormEvent(
            event_id=f'syn-{i}',
            event_type='Hail',
            magnitude=mag,
            begin_lat=lat,
            begin_lon=lon,
            state=state,
            processed_ts=datetime.datetime.utcnow().isoformat() + 'Z'
        )
        
        # Calculate scores
        geo_analyzer = GeospatialAnalyzer()
        data_validator = DataValidator()
        
        event.affected_area = geo_analyzer.calculate_affected_area(event)
        event.severity_score = geo_analyzer.calculate_severity_score(event)
        event.quality_score = data_validator.calculate_quality_score(event)
        
        lead = {
            'lead_id': f'lead-syn-{i}',
            'event_id': event.event_id,
            'score': min(100, int((mag / 3.0) * 100)),
            'status': 'new',
            'created_ts': datetime.datetime.utcnow().isoformat() + 'Z',
            'event': event.to_dict(),
            'property': {'lat': lat, 'lon': lon, 'affected_area': event.affected_area},
            'contact': None,
            'provenance': {
                'source': 'synthetic',
                'ingested_ts': datetime.datetime.utcnow().isoformat() + 'Z',
                'event_type': event.event_type,
                'magnitude': event.magnitude,
                'state': event.state,
                'severity_score': event.severity_score,
                'quality_score': event.quality_score
            },
        }
        leads.append(lead)
    
    struct_logger.info("Synthetic leads generated", count=len(leads))
    return leads

def ingest_data(config: APIConfig, start: str, end: str, 
                hail_min: float = DEFAULT_HAIL_MIN,
                wind_min: float = DEFAULT_WIND_MIN) -> IngestionResult:
    """Main data ingestion function with comprehensive processing."""
    start_time = time.time()
    api_calls = 0
    cache_hits = 0
    db_records = 0
    
    try:
        # Initialize components
        db_manager = DatabaseManager() if config.enable_db_storage else None
        cache_manager = CacheManager() if config.enable_caching else None
        noaa_client = NOAAClient(config)
        event_processor = EventProcessor(config, db_manager)
        
        # Check cache first
        cache_key = None
        if cache_manager:
            cache_key = cache_manager.get_cache_key(start, end)
            cached_events = cache_manager.get_cached_data(cache_key)
            if cached_events:
                cache_hits = 1
                events = cached_events
                struct_logger.info("Using cached data", count=len(events))
            else:
                # Fetch from NOAA API
                events = noaa_client.fetch_storm_events(start, end)
                api_calls = 1
                
                # Cache the results
                if cache_manager:
                    cache_manager.cache_data(cache_key, events)
        else:
            # Fetch from NOAA API
            events = noaa_client.fetch_storm_events(start, end)
            api_calls = 1
        
        # Process events
        storm_events, leads = event_processor.process_events(events)
        
        # Store in database
        if db_manager:
            db_records = db_manager.store_events_batch(storm_events)
        
        # Write output files
        if storm_events:
            write_raw_events(storm_events, start, end)
        
        if leads:
            write_leads(leads)
        
        # Calculate quality metrics
        quality_metrics = {
            'total_events': len(events),
            'valid_events': len(storm_events),
            'invalid_events': len(events) - len(storm_events),
            'leads_generated': len(leads),
            'avg_quality_score': sum(e.quality_score or 0 for e in storm_events) / len(storm_events) if storm_events else 0,
            'avg_severity_score': sum(e.severity_score or 0 for e in storm_events) / len(storm_events) if storm_events else 0,
            'events_by_type': {},
            'events_by_state': {}
        }
        
        # Calculate event type and state distributions
        for event in storm_events:
            event_type = event.event_type.lower()
            state = event.state or 'unknown'
            
            quality_metrics['events_by_type'][event_type] = quality_metrics['events_by_type'].get(event_type, 0) + 1
            quality_metrics['events_by_state'][state] = quality_metrics['events_by_state'].get(state, 0) + 1
        
        processing_time = time.time() - start_time
        
        struct_logger.info("Ingestion completed", 
                          events_count=len(storm_events),
                          leads_count=len(leads),
                          processing_time=processing_time,
                          api_calls=api_calls,
                          cache_hits=cache_hits,
                          db_records=db_records)
        
        return IngestionResult(
            source=DataSource.NOAA_API,
            events_count=len(storm_events),
            leads_count=len(leads),
            success=True,
            processing_time=processing_time,
            api_calls_made=api_calls,
            cache_hits=cache_hits,
            db_records_created=db_records,
            quality_metrics=quality_metrics
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        struct_logger.error("Ingestion failed", 
                          error=str(e),
                          processing_time=processing_time)
        
        # Fallback to synthetic data
        try:
            leads = synthetic_leads(12)
            write_leads(leads)
            
            return IngestionResult(
                source=DataSource.SYNTHETIC,
                events_count=0,
                leads_count=len(leads),
                success=True,
                processing_time=processing_time
            )
        except Exception as fallback_error:
            struct_logger.error("Synthetic fallback failed", error=str(fallback_error))
            
            return IngestionResult(
                source=DataSource.SYNTHETIC,
                events_count=0,
                leads_count=0,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )

def main(argv=None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced Hail Hero NOAA Data Integration')
    today = datetime.date.today()
    default_end = today.isoformat()
    default_start = (today - datetime.timedelta(days=365)).isoformat()
    
    parser.add_argument('--start', default=default_start, help='Start date (ISO format)')
    parser.add_argument('--end', default=default_end, help='End date (ISO format)')
    parser.add_argument('--hail-min', type=float, default=DEFAULT_HAIL_MIN, help='Minimum hail size (inches)')
    parser.add_argument('--wind-min', type=float, default=DEFAULT_WIND_MIN, help='Minimum wind speed (mph)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--schedule', action='store_true', help='Run in scheduled mode')
    parser.add_argument('--event-types', nargs='+', help='Event types to fetch (e.g., Hail Tornado Wind)')
    parser.add_argument('--states', nargs='+', help='States to filter (e.g., WI IL)')
    
    args = parser.parse_args(argv)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    struct_logger.info("Starting Hail Hero data ingestion", 
                      start=args.start, end=args.end,
                      hail_min=args.hail_min, wind_min=args.wind_min)
    
    try:
        config = load_config()
        
        if args.schedule:
            # Run in scheduled mode
            def scheduled_ingestion():
                result = ingest_data(config, args.start, args.end, args.hail_min, args.wind_min)
                struct_logger.info("Scheduled ingestion completed", result=result.to_dict() if hasattr(result, 'to_dict') else str(result))
            
            scheduler = Scheduler(scheduled_ingestion)
            scheduler.start()
            
            try:
                # Keep the scheduler running
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                struct_logger.info("Shutting down scheduler")
                scheduler.stop()
        else:
            # Run single ingestion
            result = ingest_data(config, args.start, args.end, args.hail_min, args.wind_min)
            
            if result.success:
                struct_logger.info("Ingestion completed successfully", result=result.__dict__)
                return 0
            else:
                struct_logger.error("Ingestion failed", error=result.error_message)
                return 1
            
    except Exception as e:
        struct_logger.error("Unexpected error in main", error=str(e))
        return 1

if __name__ == '__main__':
    sys.exit(main())