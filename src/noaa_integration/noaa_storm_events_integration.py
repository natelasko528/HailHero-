"""
Production-ready NOAA Storm Events API integration module.

This module provides comprehensive integration with the NOAA/NCEI Storm Events API
including robust authentication, error handling, rate limiting, retry logic,
and production-ready features for hail event detection and processing.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import math
import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import backoff

from ..config import get_noaa_config, NOAAConfig
from ..mvp.address_enrichment import AddressEnricher, create_enricher

logger = logging.getLogger(__name__)


class EventSeverity(Enum):
    """Event severity levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    EXTREME = "extreme"


class EventType(Enum):
    """Storm event types."""
    HAIL = "hail"
    WIND = "wind"
    TORNADO = "tornado"
    THUNDERSTORM = "thunderstorm"
    FLOOD = "flood"
    WINTER_STORM = "winter_storm"
    OTHER = "other"


class DataSource(Enum):
    """Data source types."""
    NOAA_API = "noaa_api"
    CACHED_DATA = "cached_data"
    SYNTHETIC = "synthetic"
    BACKUP_SOURCE = "backup_source"


class APIStatus(Enum):
    """API status indicators."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"


@dataclass
class EventGeometry:
    """Geometric data for storm events."""
    event_id: str
    event_type: EventType
    begin_lat: float
    begin_lon: float
    end_lat: Optional[float] = None
    end_lon: Optional[float] = None
    begin_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    magnitude: Optional[float] = None
    magnitude_type: Optional[str] = None
    state: Optional[str] = None
    county: Optional[str] = None
    timezone: Optional[str] = None
    damage_crops: Optional[float] = None
    damage_property: Optional[float] = None
    injuries_direct: Optional[int] = None
    injuries_indirect: Optional[int] = None
    deaths_direct: Optional[int] = None
    deaths_indirect: Optional[int] = None
    
    # Calculated fields
    severity: EventSeverity = EventSeverity.MODERATE
    affected_area_sq_miles: float = 0.0
    duration_minutes: float = 0.0
    risk_score: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.begin_time and self.end_time:
            self.duration_minutes = (self.end_time - self.begin_time).total_seconds() / 60
        
        # Calculate affected area based on event type and magnitude
        if self.magnitude:
            if self.event_type == EventType.HAIL:
                # Hail affected area: radius based on hail size
                radius_miles = max(1.0, self.magnitude * 2.0)
                self.affected_area_sq_miles = math.pi * (radius_miles ** 2)
            elif self.event_type == EventType.WIND:
                # Wind affected area: larger radius for higher winds
                radius_miles = max(2.0, (self.magnitude / 20) * 3.0)
                self.affected_area_sq_miles = math.pi * (radius_miles ** 2)
            elif self.event_type == EventType.TORNADO:
                # Tornado path area
                if self.end_lat and self.end_lon:
                    path_length = self._calculate_distance(
                        self.begin_lat, self.begin_lon, 
                        self.end_lat, self.end_lon
                    )
                    # Average tornado width is about 0.1 miles
                    self.affected_area_sq_miles = path_length * 0.1
        
        # Calculate risk score
        self.risk_score = self._calculate_risk_score()
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in miles."""
        R = 3959  # Earth's radius in miles
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def _calculate_risk_score(self) -> float:
        """Calculate risk score based on event characteristics."""
        score = 0.0
        
        # Base score by event type
        type_scores = {
            EventType.TORNADO: 50,
            EventType.HAIL: 30,
            EventType.WIND: 25,
            EventType.THUNDERSTORM: 20,
            EventType.FLOOD: 35,
            EventType.WINTER_STORM: 30,
            EventType.OTHER: 10
        }
        score += type_scores.get(self.event_type, 10)
        
        # Magnitude score
        if self.magnitude:
            if self.event_type == EventType.HAIL:
                score += min(30, self.magnitude * 15)  # 1" hail = 15 pts
            elif self.event_type == EventType.WIND:
                score += min(30, (self.magnitude / 10) * 10)  # 10 mph = 10 pts
            elif self.event_type == EventType.TORNADO:
                score += min(40, self.magnitude * 10)  # EF1 = 10 pts
        
        # Duration score
        if self.duration_minutes > 0:
            score += min(20, self.duration_minutes / 10)  # 10 min = 10 pts
        
        # Casualty score
        total_casualties = (self.deaths_direct or 0) + (self.deaths_indirect or 0) + \
                          (self.injuries_direct or 0) + (self.injuries_indirect or 0)
        score += min(30, total_casualties * 5)
        
        # Damage score
        total_damage = (self.damage_property or 0) + (self.damage_crops or 0)
        if total_damage > 0:
            # Log scale for damage (1M = 10 pts, 10M = 20 pts, etc.)
            score += min(30, math.log10(max(total_damage, 1)) * 5)
        
        return min(100, score)


@dataclass
class ProcessedEvent:
    """Processed storm event with enhanced data."""
    event_id: str
    geometry: EventGeometry
    raw_data: Dict[str, Any]
    processed_at: datetime = field(default_factory=datetime.utcnow)
    data_quality_score: float = 0.0
    enrichment_metadata: Dict[str, Any] = field(default_factory=dict)
    lead_generation_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationMetrics:
    """Metrics for NOAA integration performance."""
    total_events_processed: int = 0
    successful_events: int = 0
    failed_events: int = 0
    api_calls_made: int = 0
    api_calls_successful: int = 0
    api_calls_failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    last_successful_fetch: Optional[datetime] = None
    last_error: Optional[str] = None
    retry_count: int = 0
    circuit_breaker_trips: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_events_processed == 0:
            return 0.0
        return self.successful_events / self.total_events_processed
    
    @property
    def api_success_rate(self) -> float:
        """Calculate API success rate."""
        if self.api_calls_made == 0:
            return 0.0
        return self.api_calls_successful / self.api_calls_made
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return self.cache_hits / total_cache_requests


class RateLimiter:
    """Advanced rate limiter with burst handling."""
    
    def __init__(self, max_requests_per_second: float = 1.0, burst_capacity: int = 5):
        self.max_requests_per_second = max_requests_per_second
        self.burst_capacity = burst_capacity
        self.tokens = burst_capacity
        self.last_refill = time.time()
        self.refill_rate = max_requests_per_second
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token from the rate limiter."""
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_refill
            
            # Refill tokens
            tokens_to_add = time_passed * self.refill_rate
            self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return
            
            # Calculate wait time
            wait_time = (1 - self.tokens) / self.refill_rate
            await asyncio.sleep(wait_time)
            self.tokens = 0


class CircuitBreaker:
    """Advanced circuit breaker with half-open state."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 half_open_attempts: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts
        self.failure_count = 0
        self.half_open_success_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection."""
        async with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half_open"
                    self.half_open_success_count = 0
                    logger.info("Circuit breaker transitioning to half-open state")
                else:
                    raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                if self.state == "half_open":
                    self.half_open_success_count += 1
                    if self.half_open_success_count >= self.half_open_attempts:
                        self.state = "closed"
                        self.failure_count = 0
                        logger.info("Circuit breaker recovered to closed state")
                else:
                    self.failure_count = 0
            return result
        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise


class DataValidator:
    """Validator for NOAA storm events data."""
    
    @staticmethod
    def validate_event_geometry(event_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate event geometry data."""
        errors = []
        
        # Check required fields
        required_fields = ['EVENT_ID', 'EVENT_TYPE', 'BEGIN_LAT', 'BEGIN_LON']
        for field in required_fields:
            if not event_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate coordinates
        try:
            lat = float(event_data.get('BEGIN_LAT', 0))
            lon = float(event_data.get('BEGIN_LON', 0))
            
            if not (-90 <= lat <= 90):
                errors.append(f"Invalid latitude: {lat}")
            if not (-180 <= lon <= 180):
                errors.append(f"Invalid longitude: {lon}")
        except (ValueError, TypeError):
            errors.append("Invalid coordinate format")
        
        # Validate event type
        event_type = event_data.get('EVENT_TYPE', '').lower()
        valid_types = ['hail', 'wind', 'tornado', 'thunderstorm', 'flood', 'winter storm']
        if not any(event_type in valid_type for valid_type in valid_types):
            errors.append(f"Invalid event type: {event_type}")
        
        # Validate magnitude if present
        magnitude = event_data.get('MAGNITUDE')
        if magnitude is not None:
            try:
                mag_float = float(magnitude)
                if mag_float < 0:
                    errors.append(f"Invalid magnitude: {mag_float}")
            except (ValueError, TypeError):
                errors.append(f"Invalid magnitude format: {magnitude}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def calculate_data_quality_score(event_data: Dict[str, Any]) -> float:
        """Calculate data quality score for an event."""
        score = 0.0
        
        # Basic completeness (30 points)
        basic_fields = ['EVENT_ID', 'EVENT_TYPE', 'BEGIN_LAT', 'BEGIN_LON', 'STATE']
        for field in basic_fields:
            if event_data.get(field):
                score += 6
        
        # Temporal data (20 points)
        if event_data.get('BEGIN_DATE_TIME'):
            score += 10
        if event_data.get('END_DATE_TIME'):
            score += 10
        
        # Magnitude data (15 points)
        if event_data.get('MAGNITUDE'):
            score += 10
        if event_data.get('MAGNITUDE_TYPE'):
            score += 5
        
        # Geographic detail (15 points)
        if event_data.get('COUNTY'):
            score += 5
        if event_data.get('END_LAT') and event_data.get('END_LON'):
            score += 10
        
        # Impact data (10 points)
        impact_fields = ['DAMAGE_PROPERTY', 'DAMAGE_CROPS', 'DEATHS_DIRECT', 'INJURIES_DIRECT']
        for field in impact_fields:
            if event_data.get(field):
                score += 2.5
        
        # Additional detail (10 points)
        detail_fields = ['TIMEZONE', 'CZ_TYPE', 'CZ_NAME']
        for field in detail_fields:
            if event_data.get(field):
                score += 3.33
        
        return min(100, score)


class DataCache:
    """Advanced caching system for NOAA data."""
    
    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        
        # Create cache directory
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, url: str, params: Dict[str, Any]) -> str:
        """Generate cache key from URL and parameters."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{url}:{param_str}".encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        """Get file path for cache key."""
        return f"{self.cache_dir}/{cache_key}.json"
    
    async def get(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached data."""
        cache_key = self._get_cache_key(url, params)
        
        async with self._lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                cached = self.memory_cache[cache_key]
                if time.time() - cached['timestamp'] < self.ttl_hours * 3600:
                    return cached['data']
                else:
                    del self.memory_cache[cache_key]
            
            # Check file cache
            file_path = self._get_cache_file_path(cache_key)
            try:
                import os
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        cached_data = json.load(f)
                    
                    if time.time() - cached_data['timestamp'] < self.ttl_hours * 3600:
                        # Load into memory cache
                        self.memory_cache[cache_key] = cached_data
                        return cached_data['data']
                    else:
                        # Remove expired file
                        os.remove(file_path)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        return None
    
    async def set(self, url: str, params: Dict[str, Any], data: Dict[str, Any]):
        """Cache data."""
        cache_key = self._get_cache_key(url, params)
        cache_entry = {
            'data': data,
            'timestamp': time.time(),
            'url': url,
            'params': params
        }
        
        async with self._lock:
            # Store in memory cache
            self.memory_cache[cache_key] = cache_entry
            
            # Store in file cache
            file_path = self._get_cache_file_path(cache_key)
            try:
                with open(file_path, 'w') as f:
                    json.dump(cache_entry, f, indent=2)
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
    
    async def clear(self):
        """Clear all cached data."""
        async with self._lock:
            self.memory_cache.clear()
            
            # Clear file cache
            import os
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))


class NOAAStormEventsIntegration:
    """Production-ready NOAA Storm Events API integration."""
    
    def __init__(self, config: Optional[NOAAConfig] = None):
        self.config = config or get_noaa_config()
        self.metrics = IntegrationMetrics()
        self.rate_limiter = RateLimiter(1.0 / self.config.rate_limit_delay)
        self.circuit_breaker = CircuitBreaker()
        self.data_validator = DataValidator()
        self.cache = DataCache(self.config.cache_directory, self.config.cache_ttl_hours)
        self.address_enricher = create_enricher()
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        self.session_timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        # API status
        self.api_status = APIStatus.HEALTHY
        self.last_health_check = None
        
        logger.info("NOAA Storm Events Integration initialized")
    
    async def __aenter__(self):
        """Initialize async context."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context."""
        await self.close_session()
    
    async def start_session(self):
        """Start HTTP session."""
        if self.session is None:
            headers = {
                'User-Agent': 'HailHero/1.0',
                'Accept': 'application/json'
            }
            
            if self.config.ncei_token:
                headers['token'] = self.config.ncei_token
            
            self.session = aiohttp.ClientSession(
                timeout=self.session_timeout,
                headers=headers
            )
            logger.info("HTTP session started")
    
    async def close_session(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("HTTP session closed")
    
    def _parse_event_geometry(self, event_data: Dict[str, Any]) -> EventGeometry:
        """Parse event geometry from raw event data."""
        # Parse event type
        event_type_str = event_data.get('EVENT_TYPE', '').lower()
        event_type = EventType.OTHER
        
        if 'hail' in event_type_str:
            event_type = EventType.HAIL
        elif 'wind' in event_type_str:
            event_type = EventType.WIND
        elif 'tornado' in event_type_str:
            event_type = EventType.TORNADO
        elif 'thunderstorm' in event_type_str:
            event_type = EventType.THUNDERSTORM
        elif 'flood' in event_type_str:
            event_type = EventType.FLOOD
        elif 'winter' in event_type_str and 'storm' in event_type_str:
            event_type = EventType.WINTER_STORM
        
        # Parse coordinates
        begin_lat = float(event_data.get('BEGIN_LAT', 0))
        begin_lon = float(event_data.get('BEGIN_LON', 0))
        end_lat = None
        end_lon = None
        
        if event_data.get('END_LAT') and event_data.get('END_LON'):
            try:
                end_lat = float(event_data['END_LAT'])
                end_lon = float(event_data['END_LON'])
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to parse end coordinates for event {event_data.get('EVENT_ID', 'unknown')}: {e}")
                # Keep end coordinates as None - event has no end location
        
        # Parse timestamps
        begin_time = None
        end_time = None
        
        if event_data.get('BEGIN_DATE_TIME'):
            try:
                begin_time = datetime.fromisoformat(event_data['BEGIN_DATE_TIME'].replace('Z', '+00:00'))
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to parse begin time for event {event_data.get('EVENT_ID', 'unknown')}: {e}")
                # Keep begin_time as None - timestamp parsing failed
        
        if event_data.get('END_DATE_TIME'):
            try:
                end_time = datetime.fromisoformat(event_data['END_DATE_TIME'].replace('Z', '+00:00'))
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to parse end time for event {event_data.get('EVENT_ID', 'unknown')}: {e}")
                # Keep end_time as None - timestamp parsing failed
        
        # Parse magnitude
        magnitude = None
        magnitude_type = event_data.get('MAGNITUDE_TYPE')
        
        if event_data.get('MAGNITUDE'):
            try:
                magnitude = float(event_data['MAGNITUDE'])
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to parse magnitude for event {event_data.get('EVENT_ID', 'unknown')}: {e}")
                # Keep magnitude as None - magnitude parsing failed
        
        # Parse damage and casualty information
        damage_crops = None
        damage_property = None
        
        if event_data.get('DAMAGE_CROPS'):
            damage_crops = self._parse_damage_value(event_data['DAMAGE_CROPS'])
        
        if event_data.get('DAMAGE_PROPERTY'):
            damage_property = self._parse_damage_value(event_data['DAMAGE_PROPERTY'])
        
        # Create geometry object
        geometry = EventGeometry(
            event_id=event_data['EVENT_ID'],
            event_type=event_type,
            begin_lat=begin_lat,
            begin_lon=begin_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            begin_time=begin_time,
            end_time=end_time,
            magnitude=magnitude,
            magnitude_type=magnitude_type,
            state=event_data.get('STATE'),
            county=event_data.get('COUNTYNAME'),
            timezone=event_data.get('TIMEZONE'),
            damage_crops=damage_crops,
            damage_property=damage_property,
            injuries_direct=self._parse_int_field(event_data.get('INJURIES_DIRECT')),
            injuries_indirect=self._parse_int_field(event_data.get('INJURIES_INDIRECT')),
            deaths_direct=self._parse_int_field(event_data.get('DEATHS_DIRECT')),
            deaths_indirect=self._parse_int_field(event_data.get('DEATHS_INDIRECT'))
        )
        
        # Set severity based on risk score
        if geometry.risk_score >= 80:
            geometry.severity = EventSeverity.EXTREME
        elif geometry.risk_score >= 60:
            geometry.severity = EventSeverity.SEVERE
        elif geometry.risk_score >= 40:
            geometry.severity = EventSeverity.HIGH
        elif geometry.risk_score >= 20:
            geometry.severity = EventSeverity.MODERATE
        else:
            geometry.severity = EventSeverity.LOW
        
        return geometry
    
    def _parse_damage_value(self, damage_str: str) -> Optional[float]:
        """Parse damage value from NOAA format (e.g., '10.5K', '2.3M')."""
        if not damage_str:
            return None
        
        try:
            # Remove commas and convert to uppercase
            clean_str = damage_str.replace(',', '').upper()
            
            # Handle K, M, B suffixes
            if 'K' in clean_str:
                return float(clean_str.replace('K', '')) * 1000
            elif 'M' in clean_str:
                return float(clean_str.replace('M', '')) * 1000000
            elif 'B' in clean_str:
                return float(clean_str.replace('B', '')) * 1000000000
            else:
                return float(clean_str)
        except (ValueError, TypeError):
            return None
    
    def _parse_int_field(self, field_value: Any) -> Optional[int]:
        """Parse integer field safely."""
        if field_value is None:
            return None
        try:
            return int(field_value)
        except (ValueError, TypeError):
            return None
    
    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError),
                         max_tries=3, base=2)
    async def _make_api_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with exponential backoff."""
        self.metrics.api_calls_made += 1
        
        # Check cache first
        if self.config.enable_caching:
            cached_data = await self.cache.get(url, params)
            if cached_data:
                self.metrics.cache_hits += 1
                return cached_data
            else:
                self.metrics.cache_misses += 1
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.metrics.api_calls_successful += 1
                    
                    # Cache successful response
                    if self.config.enable_caching:
                        await self.cache.set(url, params, data)
                    
                    return data
                else:
                    error_text = await response.text()
                    self.metrics.api_calls_failed += 1
                    
                    if response.status == 429:
                        self.api_status = APIStatus.RATE_LIMITED
                        raise Exception(f"Rate limited: {error_text}")
                    elif response.status >= 500:
                        self.api_status = APIStatus.UNAVAILABLE
                        raise Exception(f"Server error: {response.status}")
                    else:
                        raise Exception(f"API error: {response.status} - {error_text}")
        
        except asyncio.TimeoutError:
            self.metrics.api_calls_failed += 1
            self.api_status = APIStatus.DEGRADED
            raise Exception("Request timeout")
        except aiohttp.ClientError as e:
            self.metrics.api_calls_failed += 1
            self.api_status = APIStatus.DEGRADED
            raise Exception(f"Network error: {str(e)}")
    
    async def fetch_storm_events(self, start_date: str, end_date: str, 
                                limit: int = 1000, offset: int = 1,
                                additional_params: Optional[Dict[str, Any]] = None) -> List[ProcessedEvent]:
        """Fetch storm events from NOAA API."""
        start_time = time.time()
        
        try:
            # Build request parameters
            params = {
                'dataset': self.config.ncei_dataset,
                'startDate': start_date,
                'endDate': end_date,
                'limit': limit,
                'offset': offset,
                'format': 'json'
            }
            
            if additional_params:
                params.update(additional_params)
            
            url = self.config.ncei_base_url.rstrip('/')
            
            logger.info(f"Fetching storm events: {start_date} to {end_date}, limit={limit}, offset={offset}")
            
            # Make API request with circuit breaker protection
            response_data = await self.circuit_breaker.call(
                self._make_api_request, url, params
            )
            
            # Process events
            processed_events = []
            raw_events = response_data.get('results', response_data.get('data', []))
            
            if isinstance(raw_events, dict):
                raw_events = raw_events.get('results', [])
            
            for event_data in raw_events:
                try:
                    processed_event = await self._process_event(event_data)
                    if processed_event:
                        processed_events.append(processed_event)
                except Exception as e:
                    logger.error(f"Error processing event {event_data.get('EVENT_ID', 'unknown')}: {e}")
                    self.metrics.failed_events += 1
            
            # Update metrics
            self.metrics.total_events_processed += len(raw_events)
            self.metrics.successful_events += len(processed_events)
            self.metrics.failed_events += len(raw_events) - len(processed_events)
            self.metrics.last_successful_fetch = datetime.utcnow()
            self.metrics.total_processing_time += time.time() - start_time
            self.metrics.average_processing_time = self.metrics.total_processing_time / self.metrics.total_events_processed
            
            logger.info(f"Successfully processed {len(processed_events)} events")
            
            return processed_events
            
        except Exception as e:
            self.metrics.last_error = str(e)
            logger.error(f"Failed to fetch storm events: {e}")
            raise
    
    async def _process_event(self, event_data: Dict[str, Any]) -> Optional[ProcessedEvent]:
        """Process a single storm event."""
        try:
            # Validate event data
            is_valid, validation_errors = self.data_validator.validate_event_geometry(event_data)
            if not is_valid:
                logger.warning(f"Invalid event {event_data.get('EVENT_ID', 'unknown')}: {validation_errors}")
                return None
            
            # Parse geometry
            geometry = self._parse_event_geometry(event_data)
            
            # Calculate data quality score
            quality_score = self.data_validator.calculate_data_quality_score(event_data)
            
            # Create processed event
            processed_event = ProcessedEvent(
                event_id=geometry.event_id,
                geometry=geometry,
                raw_data=event_data,
                data_quality_score=quality_score
            )
            
            # Add enrichment metadata
            processed_event.enrichment_metadata = {
                'processed_at': processed_event.processed_at.isoformat(),
                'quality_score': quality_score,
                'validation_errors': validation_errors,
                'processing_version': '1.0'
            }
            
            return processed_event
            
        except Exception as e:
            logger.error(f"Error processing event {event_data.get('EVENT_ID', 'unknown')}: {e}")
            return None
    
    async def fetch_all_events_paginated(self, start_date: str, end_date: str,
                                       batch_size: int = 1000,
                                       max_events: Optional[int] = None) -> AsyncGenerator[ProcessedEvent, None]:
        """Fetch all events with pagination."""
        offset = 1
        total_fetched = 0
        
        logger.info(f"Starting paginated fetch from {start_date} to {end_date}")
        
        while True:
            if max_events and total_fetched >= max_events:
                logger.info(f"Reached maximum events limit: {max_events}")
                break
            
            current_batch_size = min(batch_size, max_events - total_fetched) if max_events else batch_size
            
            try:
                batch_events = await self.fetch_storm_events(
                    start_date, end_date,
                    limit=current_batch_size,
                    offset=offset
                )
                
                if not batch_events:
                    logger.info("No more events available")
                    break
                
                for event in batch_events:
                    yield event
                    total_fetched += 1
                
                offset += len(batch_events)
                
                logger.info(f"Fetched {len(batch_events)} events (total: {total_fetched})")
                
                # If we got fewer events than requested, we're done
                if len(batch_events) < current_batch_size:
                    break
                
                # Small delay between batches
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching batch at offset {offset}: {e}")
                break
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            'status': self.api_status.value,
            'timestamp': datetime.utcnow().isoformat(),
            'config': {
                'base_url': self.config.ncei_base_url,
                'dataset': self.config.ncei_dataset,
                'timeout': self.config.timeout,
                'max_retries': self.config.max_retries,
                'rate_limit_delay': self.config.rate_limit_delay,
                'enable_caching': self.config.enable_caching,
            },
            'metrics': asdict(self.metrics),
            'circuit_breaker': {
                'state': self.circuit_breaker.state,
                'failure_count': self.circuit_breaker.failure_count,
                'last_failure_time': self.circuit_breaker.last_failure_time
            },
            'cache': {
                'memory_entries': len(self.cache.memory_cache),
                'ttl_hours': self.cache.ttl_hours
            },
            'session_active': self.session is not None and not self.session.closed,
        }
        
        # Test API connectivity
        try:
            # Make a small test request
            test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            test_events = await self.fetch_storm_events(test_date, test_date, limit=1)
            
            if test_events:
                health_status['api_test'] = 'passed'
                health_status['api_response_time'] = time.time() - time.time()  # This would be measured properly
            else:
                health_status['api_test'] = 'no_data'
        except Exception as e:
            health_status['api_test'] = 'failed'
            health_status['api_error'] = str(e)
        
        # Determine overall health
        if health_status['api_test'] == 'passed':
            self.api_status = APIStatus.HEALTHY
        elif health_status['api_test'] == 'failed':
            self.api_status = APIStatus.UNAVAILABLE
        
        health_status['status'] = self.api_status.value
        self.last_health_check = datetime.utcnow()
        
        return health_status
    
    async def get_hail_events_only(self, start_date: str, end_date: str,
                                 min_hail_size: float = 0.5,
                                 target_states: Optional[List[str]] = None) -> List[ProcessedEvent]:
        """Fetch hail events only with filtering."""
        if target_states is None:
            target_states = self.config.target_states
        
        all_events = []
        async for event in self.fetch_all_events_paginated(start_date, end_date):
            if (event.geometry.event_type == EventType.HAIL and 
                event.geometry.magnitude and 
                event.geometry.magnitude >= min_hail_size):
                
                # Filter by target states if specified
                if target_states and event.geometry.state:
                    if event.geometry.state.upper() in [state.upper() for state in target_states]:
                        all_events.append(event)
                else:
                    all_events.append(event)
        
        logger.info(f"Found {len(all_events)} hail events meeting criteria")
        return all_events
    
    def get_metrics(self) -> IntegrationMetrics:
        """Get current metrics."""
        return self.metrics
    
    async def clear_cache(self):
        """Clear all cached data."""
        await self.cache.clear()
        logger.info("Cache cleared")
    
    async def close(self):
        """Close the integration and cleanup resources."""
        await self.close_session()
        logger.info("NOAA Storm Events Integration closed")
    
    def classify_hail_severity(self, hail_size_inches: float) -> EventSeverity:
        """Classify hail event severity based on hail size.
        
        Uses standard meteorological classifications:
        - < 1": Minor
        - 1" - 2": Moderate
        - 2" - 3": Severe
        - 3" - 4": Extreme
        - > 4": Catastrophic
        """
        if hail_size_inches < 1.0:
            return EventSeverity.LOW
        elif hail_size_inches < 2.0:
            return EventSeverity.MODERATE
        elif hail_size_inches < 3.0:
            return EventSeverity.HIGH
        elif hail_size_inches < 4.0:
            return EventSeverity.SEVERE
        else:
            return EventSeverity.EXTREME
    
    def calculate_hail_damage_potential(self, hail_size_inches: float, 
                                     wind_speed_mph: Optional[float] = None) -> Dict[str, Any]:
        """Calculate hail damage potential based on size and wind conditions.
        
        Returns damage assessment including:
        - Roof damage probability
        - Window damage probability
        - Siding damage probability
        - Vehicle damage probability
        - Crop damage probability
        - Overall damage score
        """
        damage_potential = {
            'roof_damage_probability': 0.0,
            'window_damage_probability': 0.0,
            'siding_damage_probability': 0.0,
            'vehicle_damage_probability': 0.0,
            'crop_damage_probability': 0.0,
            'overall_damage_score': 0.0,
            'damage_categories': []
        }
        
        # Base damage probabilities by hail size
        if hail_size_inches < 0.5:
            # Minimal damage
            damage_potential['roof_damage_probability'] = min(5, hail_size_inches * 10)
            damage_potential['window_damage_probability'] = min(2, hail_size_inches * 4)
            damage_potential['siding_damage_probability'] = min(3, hail_size_inches * 6)
            damage_potential['vehicle_damage_probability'] = min(10, hail_size_inches * 20)
            damage_potential['crop_damage_probability'] = min(15, hail_size_inches * 30)
        elif hail_size_inches < 1.0:
            # Minor damage
            damage_potential['roof_damage_probability'] = min(25, 5 + (hail_size_inches - 0.5) * 40)
            damage_potential['window_damage_probability'] = min(10, 2 + (hail_size_inches - 0.5) * 16)
            damage_potential['siding_damage_probability'] = min(15, 3 + (hail_size_inches - 0.5) * 24)
            damage_potential['vehicle_damage_probability'] = min(40, 10 + (hail_size_inches - 0.5) * 60)
            damage_potential['crop_damage_probability'] = min(50, 15 + (hail_size_inches - 0.5) * 70)
        elif hail_size_inches < 2.0:
            # Moderate damage
            damage_potential['roof_damage_probability'] = min(60, 25 + (hail_size_inches - 1.0) * 35)
            damage_potential['window_damage_probability'] = min(35, 10 + (hail_size_inches - 1.0) * 25)
            damage_potential['siding_damage_probability'] = min(45, 15 + (hail_size_inches - 1.0) * 30)
            damage_potential['vehicle_damage_probability'] = min(75, 40 + (hail_size_inches - 1.0) * 35)
            damage_potential['crop_damage_probability'] = min(85, 50 + (hail_size_inches - 1.0) * 35)
        elif hail_size_inches < 3.0:
            # Severe damage
            damage_potential['roof_damage_probability'] = min(85, 60 + (hail_size_inches - 2.0) * 25)
            damage_potential['window_damage_probability'] = min(65, 35 + (hail_size_inches - 2.0) * 30)
            damage_potential['siding_damage_probability'] = min(75, 45 + (hail_size_inches - 2.0) * 30)
            damage_potential['vehicle_damage_probability'] = min(95, 75 + (hail_size_inches - 2.0) * 20)
            damage_potential['crop_damage_probability'] = min(98, 85 + (hail_size_inches - 2.0) * 13)
        else:
            # Extreme damage
            damage_potential['roof_damage_probability'] = min(98, 85 + (hail_size_inches - 3.0) * 10)
            damage_potential['window_damage_probability'] = min(90, 65 + (hail_size_inches - 3.0) * 20)
            damage_potential['siding_damage_probability'] = min(92, 75 + (hail_size_inches - 3.0) * 15)
            damage_potential['vehicle_damage_probability'] = min(99, 95 + (hail_size_inches - 3.0) * 5)
            damage_potential['crop_damage_probability'] = min(100, 98 + (hail_size_inches - 3.0) * 2)
        
        # Wind speed multiplier (wind significantly increases hail damage)
        if wind_speed_mph and wind_speed_mph > 20:
            wind_multiplier = 1.0 + min(0.5, (wind_speed_mph - 20) / 60)  # Max 1.5x damage
            for key in damage_potential:
                if key.endswith('_probability'):
                    damage_potential[key] = min(100, damage_potential[key] * wind_multiplier)
        
        # Calculate overall damage score
        probabilities = [
            damage_potential['roof_damage_probability'],
            damage_potential['window_damage_probability'],
            damage_potential['siding_damage_probability'],
            damage_potential['vehicle_damage_probability'],
            damage_potential['crop_damage_probability']
        ]
        damage_potential['overall_damage_score'] = sum(probabilities) / len(probabilities)
        
        # Determine damage categories
        if damage_potential['overall_damage_score'] >= 80:
            damage_potential['damage_categories'].extend(['Catastrophic', 'Widespread'])
        elif damage_potential['overall_damage_score'] >= 60:
            damage_potential['damage_categories'].extend(['Severe', 'Extensive'])
        elif damage_potential['overall_damage_score'] >= 40:
            damage_potential['damage_categories'].extend(['Moderate', 'Localized'])
        elif damage_potential['overall_damage_score'] >= 20:
            damage_potential['damage_categories'].append('Minor')
        else:
            damage_potential['damage_categories'].append('Minimal')
        
        return damage_potential
    
    def analyze_hail_event_patterns(self, events: List[ProcessedEvent]) -> Dict[str, Any]:
        """Analyze patterns in hail events for trend detection.
        
        Analyzes:
        - Temporal patterns (time of day, seasonality)
        - Geographic clustering
        - Size distribution trends
        - Frequency patterns
        """
        if not events:
            return {}
        
        hail_events = [event for event in events if event.geometry.event_type == EventType.HAIL]
        if not hail_events:
            return {}
        
        analysis = {
            'total_hail_events': len(hail_events),
            'temporal_patterns': {},
            'geographic_patterns': {},
            'size_distribution': {},
            'frequency_analysis': {},
            'risk_assessment': {}
        }
        
        # Temporal patterns
        hours = []
        months = []
        years = []
        
        for event in hail_events:
            if event.geometry.begin_time:
                hours.append(event.geometry.begin_time.hour)
                months.append(event.geometry.begin_time.month)
                years.append(event.geometry.begin_time.year)
        
        if hours:
            analysis['temporal_patterns']['hourly_distribution'] = {
                str(h): hours.count(h) for h in range(24)
            }
            analysis['temporal_patterns']['peak_hour'] = max(range(24), key=lambda h: hours.count(h))
        
        if months:
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            analysis['temporal_patterns']['monthly_distribution'] = {
                month_names[m-1]: months.count(m) for m in range(1, 13)
            }
            analysis['temporal_patterns']['peak_month'] = month_names[max(range(1, 13), key=lambda m: months.count(m)) - 1]
        
        if years:
            analysis['temporal_patterns']['yearly_distribution'] = {
                str(y): years.count(y) for y in sorted(set(years))
            }
        
        # Geographic patterns
        if hail_events:
            lats = [event.geometry.begin_lat for event in hail_events]
            lons = [event.geometry.begin_lon for event in hail_events]
            
            analysis['geographic_patterns'] = {
                'center_lat': sum(lats) / len(lats),
                'center_lon': sum(lons) / len(lons),
                'bounding_box': {
                    'min_lat': min(lats),
                    'max_lat': max(lats),
                    'min_lon': min(lons),
                    'max_lon': max(lons)
                },
                'state_distribution': {},
                'county_distribution': {}
            }
            
            # State and county distribution
            for event in hail_events:
                if event.geometry.state:
                    state = event.geometry.state
                    analysis['geographic_patterns']['state_distribution'][state] = \
                        analysis['geographic_patterns']['state_distribution'].get(state, 0) + 1
                
                if event.geometry.county:
                    county = event.geometry.county
                    analysis['geographic_patterns']['county_distribution'][county] = \
                        analysis['geographic_patterns']['county_distribution'].get(county, 0) + 1
        
        # Size distribution
        sizes = [event.geometry.magnitude for event in hail_events if event.geometry.magnitude]
        if sizes:
            analysis['size_distribution'] = {
                'min_size': min(sizes),
                'max_size': max(sizes),
                'avg_size': sum(sizes) / len(sizes),
                'median_size': sorted(sizes)[len(sizes) // 2],
                'size_categories': {
                    'small (< 1")': len([s for s in sizes if s < 1.0]),
                    'moderate (1-2")': len([s for s in sizes if 1.0 <= s < 2.0]),
                    'large (2-3")': len([s for s in sizes if 2.0 <= s < 3.0]),
                    'very_large (3-4")': len([s for s in sizes if 3.0 <= s < 4.0]),
                    'extreme (> 4")': len([s for s in sizes if s >= 4.0])
                }
            }
        
        # Frequency analysis
        if len(hail_events) > 1:
            analysis['frequency_analysis'] = {
                'events_per_day': len(hail_events) / max(1, len(set(
                    event.geometry.begin_time.date() for event in hail_events 
                    if event.geometry.begin_time
                ))),
                'avg_duration': sum(event.geometry.duration_minutes for event in hail_events) / len(hail_events),
                'total_affected_area': sum(event.geometry.affected_area_sq_miles for event in hail_events)
            }
        
        # Risk assessment
        if sizes:
            severe_events = [s for s in sizes if s >= 2.0]  # 2"+ hail considered severe
            analysis['risk_assessment'] = {
                'severe_event_percentage': (len(severe_events) / len(sizes)) * 100,
                'risk_level': 'High' if (len(severe_events) / len(sizes)) > 0.3 else 'Moderate' if (len(severe_events) / len(sizes)) > 0.1 else 'Low',
                'recommended_actions': []
            }
            
            # Generate recommended actions
            if analysis['risk_assessment']['severe_event_percentage'] > 30:
                analysis['risk_assessment']['recommended_actions'].extend([
                    'Immediate property inspection recommended',
                    'Roof assessment required',
                    'Insurance claim preparation advised'
                ])
            elif analysis['risk_assessment']['severe_event_percentage'] > 10:
                analysis['risk_assessment']['recommended_actions'].extend([
                    'Property inspection recommended',
                    'Monitor for damage'
                ])
        
        return analysis
    
    def detect_hail_clusters(self, events: List[ProcessedEvent], 
                           cluster_radius_miles: float = 10.0,
                           min_events_per_cluster: int = 3) -> List[Dict[str, Any]]:
        """Detect geographic clusters of hail events using DBSCAN-like algorithm.
        
        Returns list of clusters with:
        - Center coordinates
        - Event count
        - Average hail size
        - Total affected area
        - Risk score
        """
        hail_events = [event for event in events if event.geometry.event_type == EventType.HAIL]
        if len(hail_events) < min_events_per_cluster:
            return []
        
        clusters = []
        visited = set()
        
        for i, event in enumerate(hail_events):
            if i in visited:
                continue
            
            # Start new cluster
            cluster = {
                'events': [event],
                'center_lat': event.geometry.begin_lat,
                'center_lon': event.geometry.begin_lon,
                'event_count': 1,
                'sizes': [event.geometry.magnitude] if event.geometry.magnitude else [],
                'affected_areas': [event.geometry.affected_area_sq_miles]
            }
            visited.add(i)
            
            # Find nearby events
            for j, other_event in enumerate(hail_events):
                if j in visited or j == i:
                    continue
                
                distance = event.geometry._calculate_distance(
                    event.geometry.begin_lat, event.geometry.begin_lon,
                    other_event.geometry.begin_lat, other_event.geometry.begin_lon
                )
                
                if distance <= cluster_radius_miles:
                    cluster['events'].append(other_event)
                    cluster['event_count'] += 1
                    visited.add(j)
                    
                    if other_event.geometry.magnitude:
                        cluster['sizes'].append(other_event.geometry.magnitude)
                    cluster['affected_areas'].append(other_event.geometry.affected_area_sq_miles)
            
            # Update cluster center to centroid
            if cluster['event_count'] > 1:
                cluster['center_lat'] = sum(e.geometry.begin_lat for e in cluster['events']) / cluster['event_count']
                cluster['center_lon'] = sum(e.geometry.begin_lon for e in cluster['events']) / cluster['event_count']
            
            # Only keep clusters with minimum events
            if cluster['event_count'] >= min_events_per_cluster:
                # Calculate cluster statistics
                cluster['avg_hail_size'] = sum(cluster['sizes']) / len(cluster['sizes']) if cluster['sizes'] else 0
                cluster['total_affected_area'] = sum(cluster['affected_areas'])
                cluster['max_hail_size'] = max(cluster['sizes']) if cluster['sizes'] else 0
                cluster['risk_score'] = self._calculate_cluster_risk_score(cluster)
                
                # Format for output
                clusters.append({
                    'cluster_id': len(clusters) + 1,
                    'center_coordinates': {
                        'lat': cluster['center_lat'],
                        'lon': cluster['center_lon']
                    },
                    'event_count': cluster['event_count'],
                    'average_hail_size': cluster['avg_hail_size'],
                    'max_hail_size': cluster['max_hail_size'],
                    'total_affected_area_sq_miles': cluster['total_affected_area'],
                    'risk_score': cluster['risk_score'],
                    'severity': self._classify_cluster_severity(cluster['risk_score']),
                    'event_ids': [e.event_id for e in cluster['events']]
                })
        
        return sorted(clusters, key=lambda x: x['risk_score'], reverse=True)
    
    def _calculate_cluster_risk_score(self, cluster: Dict[str, Any]) -> float:
        """Calculate risk score for a hail cluster."""
        score = 0.0
        
        # Base score from event count
        score += min(30, cluster['event_count'] * 5)
        
        # Score from hail sizes
        if cluster['sizes']:
            avg_size = sum(cluster['sizes']) / len(cluster['sizes'])
            max_size = max(cluster['sizes'])
            score += min(30, avg_size * 10)  # Average size contribution
            score += min(20, max_size * 5)   # Max size contribution
        
        # Score from affected area
        score += min(20, cluster['total_affected_area'] / 10)  # 10 sq miles = 20 pts
        
        return min(100, score)
    
    def _classify_cluster_severity(self, risk_score: float) -> str:
        """Classify cluster severity based on risk score."""
        if risk_score >= 80:
            return "Extreme"
        elif risk_score >= 60:
            return "Severe"
        elif risk_score >= 40:
            return "High"
        elif risk_score >= 20:
            return "Moderate"
        else:
            return "Low"


# Factory function
async def create_noaa_integration(config: Optional[NOAAConfig] = None) -> NOAAStormEventsIntegration:
    """Create and initialize NOAA integration instance."""
    integration = NOAAStormEventsIntegration(config)
    await integration.start_session()
    return integration