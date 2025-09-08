#!/usr/bin/env python3
"""
Enhanced OpenAddresses API Integration for Hail Hero

This module provides comprehensive integration with OpenAddresses API for:
- Address data retrieval and validation
- Property-level information enrichment
- Batch processing capabilities
- Advanced caching and rate limiting
- Error handling and fallback mechanisms
- Data quality scoring and confidence levels

Author: Hail Hero Development Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote, urlencode

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
OPENADDRESSES_BASE_URL = "https://openaddresses.io"
OPENADDRESSES_API_URL = "https://api.openaddresses.io/v1"
OPENADDRESSES_DATA_URL = "https://data.openaddresses.io/runs"
RATE_LIMIT_DELAY = 1.0  # seconds between requests
CACHE_TTL = 86400  # 24 hours
MAX_RETRIES = 3
BATCH_SIZE = 100

class DataSource(Enum):
    """Data source enumeration."""
    OPENADDRESSES = "openaddresses"
    COUNTY_ASSESSOR = "county_assessor"
    CENSUS = "census"
    POSTAL = "postal"
    PARCEL = "parcel"
    UNKNOWN = "unknown"

class AddressQuality(Enum):
    """Address quality levels."""
    HIGH = "high"      # Verified, complete address
    MEDIUM = "medium"  # Partial address, likely accurate
    LOW = "low"        # Incomplete or estimated address
    UNKNOWN = "unknown"

class MatchType(Enum):
    """Address match types."""
    EXACT = "exact"        # Perfect match
    FUZZY = "fuzzy"        # Close match with minor differences
    APPROXIMATE = "approximate"  # General area match
    NONE = "none"          # No match found

@dataclass
class AddressComponents:
    """Standardized address components."""
    street_number: str = ""
    street_name: str = ""
    street_type: str = ""
    street_direction: str = ""
    unit_number: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    zip_plus_four: str = ""
    county: str = ""
    country: str = "USA"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_string(self) -> str:
        """Convert address components to formatted string."""
        parts = []
        
        if self.street_number:
            parts.append(self.street_number)
        
        if self.street_direction:
            parts.append(self.street_direction)
        
        if self.street_name:
            parts.append(self.street_name)
        
        if self.street_type:
            parts.append(self.street_type)
        
        street = " ".join(parts)
        
        if self.unit_number:
            street += f" {self.unit_number}"
        
        city_line = []
        if self.city:
            city_line.append(self.city)
        
        if self.state:
            city_line.append(self.state)
        
        if self.zip_code:
            city_line.append(self.zip_code)
        
        if city_line:
            return f"{street}, {', '.join(city_line)}"
        
        return street

@dataclass
class OpenAddressesResult:
    """OpenAddresses API result."""
    address_components: AddressComponents
    original_address: str
    matched_address: str
    match_type: MatchType
    confidence_score: float
    data_source: DataSource
    source_id: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['address_components'] = self.address_components.to_dict()
        result['match_type'] = self.match_type.value
        result['data_source'] = self.data_source.value
        return result

class RateLimiter:
    """Rate limiting for API calls."""
    
    def __init__(self, delay: float = RATE_LIMIT_DELAY):
        self.delay = delay
        self.last_call = 0
        self.lock = asyncio.Lock() if asyncio else None
    
    async def async_wait(self):
        """Async wait for rate limit."""
        if self.lock:
            async with self.lock:
                await self._wait()
        else:
            self._wait()
    
    def _wait(self):
        """Wait for rate limit."""
        current_time = time.time()
        if current_time - self.last_call < self.delay:
            time.sleep(self.delay - (current_time - self.last_call))
        self.last_call = time.time()

class OpenAddressesCache:
    """Caching system for OpenAddresses results."""
    
    def __init__(self, cache_dir: Path = Path("cache/openaddresses")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite cache
        self.db_path = cache_dir / "openaddresses_cache.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize cache database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS address_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_data TEXT,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_cache_expires 
                ON address_cache(expires_at)
            ''')
            
            conn.commit()
    
    def _generate_cache_key(self, address: str, lat: Optional[float] = None, 
                          lon: Optional[float] = None) -> str:
        """Generate cache key."""
        key_parts = [address.lower().strip()]
        if lat is not None:
            key_parts.append(f"{lat:.6f}")
        if lon is not None:
            key_parts.append(f"{lon:.6f}")
        
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def get(self, address: str, lat: Optional[float] = None, 
           lon: Optional[float] = None) -> Optional[OpenAddressesResult]:
        """Get cached result."""
        cache_key = self._generate_cache_key(address, lat, lon)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT result_data, expires_at 
                    FROM address_cache 
                    WHERE cache_key = ? AND expires_at > datetime('now')
                ''', (cache_key,))
                
                row = cursor.fetchone()
                if row:
                    result_data = json.loads(row[0])
                    return self._deserialize_result(result_data)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    def set(self, address: str, result: OpenAddressesResult, 
           lat: Optional[float] = None, lon: Optional[float] = None,
           ttl: int = CACHE_TTL):
        """Set cached result."""
        cache_key = self._generate_cache_key(address, lat, lon)
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO address_cache 
                    (cache_key, result_data, created_at, expires_at)
                    VALUES (?, ?, ?, ?)
                ''', (
                    cache_key,
                    json.dumps(result.to_dict()),
                    datetime.utcnow().isoformat(),
                    expires_at.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def _deserialize_result(self, data: Dict[str, Any]) -> OpenAddressesResult:
        """Deserialize result from cache."""
        address_components = AddressComponents(**data['address_components'])
        
        return OpenAddressesResult(
            address_components=address_components,
            original_address=data['original_address'],
            matched_address=data['matched_address'],
            match_type=MatchType(data['match_type']),
            confidence_score=data['confidence_score'],
            data_source=DataSource(data['data_source']),
            source_id=data.get('source_id', ''),
            raw_data=data.get('raw_data', {}),
            metadata=data.get('metadata', {})
        )
    
    def cleanup_expired(self):
        """Clean up expired cache entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM address_cache 
                    WHERE expires_at <= datetime('now')
                ''')
                conn.commit()
                logger.info(f"Cleaned up {cursor.rowcount} expired cache entries")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

class OpenAddressesClient:
    """Enhanced OpenAddresses API client."""
    
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: str = OPENADDRESSES_API_URL,
                 cache_dir: Optional[Path] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.cache = OpenAddressesCache(cache_dir or Path("cache/openaddresses"))
        self.rate_limiter = RateLimiter()
        
        # HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HailHero-OpenAddresses/1.0',
            'Accept': 'application/json'
        })
        
        if self.api_key:
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'
    
    async def search_address(self, address: str, 
                           city: Optional[str] = None,
                           state: Optional[str] = None,
                           zip_code: Optional[str] = None) -> OpenAddressesResult:
        """Search for address using OpenAddresses API."""
        
        # Check cache first
        cached_result = self.cache.get(address)
        if cached_result:
            logger.debug(f"Cache hit for address: {address}")
            return cached_result
        
        # Rate limiting
        await self.rate_limiter.async_wait()
        
        try:
            # Build search query
            query_parts = [address]
            if city:
                query_parts.append(city)
            if state:
                query_parts.append(state)
            if zip_code:
                query_parts.append(zip_code)
            
            query = " ".join(query_parts)
            
            # Try OpenAddresses API first
            result = await self._search_openaddresses_api(query, address)
            
            # If no result, try fallback to Nominatim
            if result.match_type == MatchType.NONE:
                logger.info(f"OpenAddresses no match, trying Nominatim fallback for: {address}")
                result = await self._search_nominatim_fallback(query, address)
            
            # Cache result if successful
            if result.match_type != MatchType.NONE:
                self.cache.set(address, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Address search error for {address}: {e}")
            return self._create_error_result(address, str(e))
    
    async def _search_openaddresses_api(self, query: str, original_address: str) -> OpenAddressesResult:
        """Search using OpenAddresses API."""
        try:
            # OpenAddresses API v1 search endpoint
            params = {
                'q': query,
                'format': 'json',
                'limit': 5,
                'countrycodes': 'us'
            }
            
            url = f"{self.base_url}/search"
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 404:
                logger.debug(f"OpenAddresses API not available for query: {query}")
                return self._create_no_match_result(original_address)
            
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                return self._process_search_results(original_address, data)
            else:
                return self._create_no_match_result(original_address)
                
        except Exception as e:
            logger.warning(f"OpenAddresses API search failed: {e}")
            return self._create_no_match_result(original_address)
    
    async def _search_nominatim_fallback(self, query: str, original_address: str) -> OpenAddressesResult:
        """Fallback search using Nominatim API."""
        try:
            params = {
                'q': query,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1,
                'countrycodes': 'us'
            }
            
            url = "https://nominatim.openstreetmap.org/search"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                # Convert Nominatim result to OpenAddresses format
                return self._process_nominatim_result(original_address, data[0])
            else:
                return self._create_no_match_result(original_address)
                
        except Exception as e:
            logger.warning(f"Nominatim fallback search failed: {e}")
            return self._create_no_match_result(original_address)
    
    def _process_nominatim_result(self, original_address: str, nominatim_data: Dict[str, Any]) -> OpenAddressesResult:
        """Process Nominatim API result."""
        address_data = nominatim_data.get('address', {})
        
        # Parse address components
        components = AddressComponents(
            street_number=address_data.get('housenumber', ''),
            street_name=address_data.get('road', ''),
            street_type=address_data.get('street_type', ''),
            street_direction=address_data.get('street_direction', ''),
            unit_number=address_data.get('unit', ''),
            city=address_data.get('city', address_data.get('town', address_data.get('village', ''))),
            state=address_data.get('state', ''),
            zip_code=address_data.get('postcode', ''),
            county=address_data.get('county', ''),
            country=address_data.get('country', 'USA'),
            latitude=float(nominatim_data.get('lat', 0)) if nominatim_data.get('lat') else None,
            longitude=float(nominatim_data.get('lon', 0)) if nominatim_data.get('lon') else None
        )
        
        # Calculate confidence score
        confidence = self._calculate_nominatim_confidence(original_address, nominatim_data)
        
        # Determine match type
        match_type = self._determine_match_type(confidence)
        
        return OpenAddressesResult(
            address_components=components,
            original_address=original_address,
            matched_address=nominatim_data.get('display_name', ''),
            match_type=match_type,
            confidence_score=confidence,
            data_source=DataSource.OPENSTREETMAP,
            source_id=nominatim_data.get('place_id', ''),
            raw_data=nominatim_data,
            metadata={
                'api_version': 'nominatim',
                'importance': nominatim_data.get('importance', 0),
                'search_timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def _calculate_nominatim_confidence(self, original_address: str, nominatim_data: Dict[str, Any]) -> float:
        """Calculate confidence score for Nominatim result."""
        score = 0.0
        
        # Base score for having a match
        score += 30.0
        
        # Score for address completeness
        address_data = nominatim_data.get('address', {})
        if address_data.get('housenumber'):
            score += 15.0
        if address_data.get('road'):
            score += 15.0
        if address_data.get('city'):
            score += 10.0
        if address_data.get('state'):
            score += 5.0
        if address_data.get('postcode'):
            score += 5.0
        
        # Score for coordinates
        if nominatim_data.get('lat') and nominatim_data.get('lon'):
            score += 10.0
        
        # Score for importance/ranking
        importance = nominatim_data.get('importance', 0)
        score += min(importance * 100, 10.0)
        
        # Score for address type
        address_type = nominatim_data.get('type', '')
        if address_type in ['house', 'residential', 'building']:
            score += 10.0
        elif address_type in ['road', 'street']:
            score += 5.0
        
        return min(score, 100.0)
    
    async def search_by_coordinates(self, latitude: float, longitude: float,
                                   radius_km: float = 0.1) -> List[OpenAddressesResult]:
        """Search for addresses near coordinates."""
        
        # Rate limiting
        await self.rate_limiter.async_wait()
        
        try:
            # Build search parameters
            params = {
                'lat': latitude,
                'lon': longitude,
                'radius': radius_km,
                'format': 'json',
                'limit': 10
            }
            
            url = f"{self.base_url}/reverse"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for item in data:
                result = self._process_reverse_geocoding_result(latitude, longitude, item)
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"OpenAddresses reverse geocoding error: {e}")
            return []
    
    def _process_search_results(self, original_address: str, 
                               data: List[Dict[str, Any]]) -> OpenAddressesResult:
        """Process search results from API."""
        best_match = data[0]  # Take the first/best match
        
        # Parse address components
        components = self._parse_address_components(best_match)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(original_address, best_match)
        
        # Determine match type
        match_type = self._determine_match_type(confidence)
        
        return OpenAddressesResult(
            address_components=components,
            original_address=original_address,
            matched_address=best_match.get('display_name', ''),
            match_type=match_type,
            confidence_score=confidence,
            data_source=DataSource.OPENADDRESSES,
            source_id=best_match.get('id', ''),
            raw_data=best_match,
            metadata={
                'api_version': 'v1',
                'total_results': len(data),
                'search_timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def _process_reverse_geocoding_result(self, lat: float, lon: float,
                                        item: Dict[str, Any]) -> OpenAddressesResult:
        """Process reverse geocoding result."""
        components = self._parse_address_components(item)
        
        return OpenAddressesResult(
            address_components=components,
            original_address=f"{lat}, {lon}",
            matched_address=item.get('display_name', ''),
            match_type=MatchType.APPROXIMATE,
            confidence_score=0.7,  # Moderate confidence for reverse geocoding
            data_source=DataSource.OPENADDRESSES,
            source_id=item.get('id', ''),
            raw_data=item,
            metadata={
                'api_version': 'v1',
                'reverse_geocoding': True,
                'search_timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def _parse_address_components(self, data: Dict[str, Any]) -> AddressComponents:
        """Parse address components from API response."""
        address_data = data.get('address', {})
        
        return AddressComponents(
            street_number=address_data.get('housenumber', ''),
            street_name=address_data.get('road', ''),
            street_type=address_data.get('street_type', ''),
            street_direction=address_data.get('street_direction', ''),
            unit_number=address_data.get('unit', ''),
            city=address_data.get('city', address_data.get('town', address_data.get('village', ''))),
            state=address_data.get('state', ''),
            zip_code=address_data.get('postcode', ''),
            county=address_data.get('county', ''),
            country=address_data.get('country', 'USA'),
            latitude=float(data.get('lat', 0)) if data.get('lat') else None,
            longitude=float(data.get('lon', 0)) if data.get('lon') else None
        )
    
    def _calculate_confidence_score(self, original_address: str, 
                                  match_data: Dict[str, Any]) -> float:
        """Calculate confidence score for match."""
        score = 0.0
        
        # Base score for having a match
        score += 30.0
        
        # Score for address completeness
        address_data = match_data.get('address', {})
        if address_data.get('housenumber'):
            score += 15.0
        if address_data.get('road'):
            score += 15.0
        if address_data.get('city'):
            score += 10.0
        if address_data.get('state'):
            score += 5.0
        if address_data.get('postcode'):
            score += 5.0
        
        # Score for coordinates
        if match_data.get('lat') and match_data.get('lon'):
            score += 10.0
        
        # Score for importance/ranking
        importance = match_data.get('importance', 0)
        score += min(importance * 100, 10.0)
        
        return min(score, 100.0)
    
    def _determine_match_type(self, confidence_score: float) -> MatchType:
        """Determine match type based on confidence score."""
        if confidence_score >= 90.0:
            return MatchType.EXACT
        elif confidence_score >= 70.0:
            return MatchType.FUZZY
        elif confidence_score >= 50.0:
            return MatchType.APPROXIMATE
        else:
            return MatchType.NONE
    
    def _create_no_match_result(self, address: str) -> OpenAddressesResult:
        """Create result for no match found."""
        return OpenAddressesResult(
            address_components=AddressComponents(),
            original_address=address,
            matched_address="",
            match_type=MatchType.NONE,
            confidence_score=0.0,
            data_source=DataSource.UNKNOWN,
            metadata={
                'error': 'No match found',
                'search_timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def _create_error_result(self, address: str, error_message: str) -> OpenAddressesResult:
        """Create result for API error."""
        return OpenAddressesResult(
            address_components=AddressComponents(),
            original_address=address,
            matched_address="",
            match_type=MatchType.NONE,
            confidence_score=0.0,
            data_source=DataSource.UNKNOWN,
            metadata={
                'error': error_message,
                'search_timestamp': datetime.utcnow().isoformat()
            }
        )
    
    async def batch_search(self, addresses: List[str], 
                          max_workers: int = 4) -> List[OpenAddressesResult]:
        """Batch search multiple addresses."""
        results = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for address in addresses:
                task = self.search_address(address)
                tasks.append(task)
            
            # Process in batches to avoid overwhelming the API
            for i in range(0, len(tasks), max_workers):
                batch_tasks = tasks[i:i + max_workers]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch search error: {result}")
                        # Create error result for failed addresses
                        original_address = addresses[results.index(result)]
                        error_result = self._create_error_result(original_address, str(result))
                        results.append(error_result)
                    else:
                        results.append(result)
                
                # Rate limiting between batches
                if i + max_workers < len(tasks):
                    await asyncio.sleep(RATE_LIMIT_DELAY)
        
        return results
    
    def get_data_sources(self) -> List[Dict[str, Any]]:
        """Get available OpenAddresses data sources."""
        try:
            url = f"{self.base_url}/sources"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Error fetching data sources: {e}")
            return []
    
    def get_coverage_info(self, state: str, county: Optional[str] = None) -> Dict[str, Any]:
        """Get coverage information for a region."""
        try:
            params = {
                'state': state.upper(),
                'format': 'json'
            }
            
            if county:
                params['county'] = county.upper()
            
            url = f"{self.base_url}/coverage"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Error fetching coverage info: {e}")
            return {}
    
    def download_county_data(self, state: str, county: str, 
                          output_path: Optional[Path] = None) -> Path:
        """Download OpenAddresses data for a specific county."""
        try:
            # Get available data runs
            url = f"{OPENADDRESSES_DATA_URL}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            runs = response.json()
            
            # Find the latest run for the specified state/county
            target_county = f"{state.upper()}/{county.upper()}"
            
            for run in runs:
                if target_county in run.get('path', ''):
                    # Download the data
                    download_url = run.get('download_url')
                    if download_url:
                        if not output_path:
                            output_path = Path(f"data/openaddresses/{state}_{county}.csv")
                        
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Download and save the file
                        response = self.session.get(download_url, stream=True, timeout=30)
                        response.raise_for_status()
                        
                        with open(output_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        logger.info(f"Downloaded county data to {output_path}")
                        return output_path
            
            raise ValueError(f"No data found for {state}, {county}")
        
        except Exception as e:
            logger.error(f"Error downloading county data: {e}")
            raise

class AddressValidator:
    """Address validation and standardization."""
    
    def __init__(self):
        self.street_types = {
            'ST': 'STREET', 'STREET': 'STREET',
            'AVE': 'AVENUE', 'AVENUE': 'AVENUE',
            'BLVD': 'BOULEVARD', 'BOULEVARD': 'BOULEVARD',
            'DR': 'DRIVE', 'DRIVE': 'DRIVE',
            'RD': 'ROAD', 'ROAD': 'ROAD',
            'LN': 'LANE', 'LANE': 'LANE',
            'CT': 'COURT', 'COURT': 'COURT',
            'PL': 'PLACE', 'PLACE': 'PLACE',
            'WAY': 'WAY', 'WAY': 'WAY',
            'TR': 'TRAIL', 'TRAIL': 'TRAIL',
            'CIR': 'CIRCLE', 'CIRCLE': 'CIRCLE'
        }
        
        self.street_directions = {
            'N': 'NORTH', 'S': 'SOUTH', 'E': 'EAST', 'W': 'WEST',
            'NE': 'NORTHEAST', 'NW': 'NORTHWEST', 'SE': 'SOUTHEAST', 'SW': 'SOUTHWEST'
        }
        
        self.states = {
            'WI': 'WISCONSIN', 'IL': 'ILLINOIS', 'MN': 'MINNESOTA', 'IA': 'IOWA',
            'MI': 'MICHIGAN', 'IN': 'INDIANA', 'OH': 'OHIO', 'MO': 'MISSOURI'
        }
    
    def validate_address(self, address: str) -> Tuple[bool, List[str]]:
        """Validate address format and completeness."""
        errors = []
        
        if not address or not address.strip():
            errors.append("Address is empty")
            return False, errors
        
        # Check for basic address components
        has_street_number = bool(re.search(r'^\d+', address))
        has_street_name = bool(re.search(r'[a-zA-Z]{2,}', address))
        
        if not has_street_number:
            errors.append("Missing street number")
        
        if not has_street_name:
            errors.append("Missing street name")
        
        # Check for city, state, or zip
        has_city = bool(re.search(r',\s*[a-zA-Z]+', address))
        has_state = bool(re.search(r'\b[A-Z]{2}\b', address))
        has_zip = bool(re.search(r'\b\d{5}(?:-\d{4})?\b', address))
        
        if not (has_city or has_state or has_zip):
            errors.append("Missing city, state, or zip code")
        
        return len(errors) == 0, errors
    
    def standardize_address(self, address: str) -> str:
        """Standardize address format."""
        if not address:
            return ""
        
        # Convert to uppercase
        standardized = address.upper()
        
        # Standardize street types
        for abbr, full in self.street_types.items():
            standardized = re.sub(rf'\b{abbr}\b', full, standardized)
        
        # Standardize directions
        for abbr, full in self.street_directions.items():
            standardized = re.sub(rf'\b{abbr}\b', full, standardized)
        
        # Clean up extra whitespace
        standardized = re.sub(r'\s+', ' ', standardized).strip()
        
        # Standardize commas
        standardized = re.sub(r'\s*,\s*', ', ', standardized)
        
        return standardized
    
    def parse_address(self, address: str) -> AddressComponents:
        """Parse address into components."""
        components = AddressComponents()
        
        # Standardize first
        standardized = self.standardize_address(address)
        
        # Extract street number
        street_num_match = re.match(r'^(\d+)', standardized)
        if street_num_match:
            components.street_number = street_num_match.group(1)
            standardized = standardized[street_num_match.end():].strip()
        
        # Extract street direction
        for direction in self.street_directions.values():
            if standardized.startswith(direction + ' '):
                components.street_direction = direction
                standardized = standardized[len(direction) + 1:].strip()
                break
        
        # Extract street type
        for street_type in self.street_types.values():
            if standardized.endswith(' ' + street_type):
                components.street_type = street_type
                standardized = standardized[:-len(street_type) - 1].strip()
                break
        
        # Remaining is street name
        components.street_name = standardized.strip()
        
        # Try to extract city, state, zip from original address
        city_state_zip = re.search(r',\s*([^,]+)$', address)
        if city_state_zip:
            city_state_part = city_state_zip.group(1)
            
            # Extract zip code
            zip_match = re.search(r'\b(\d{5})(?:-(\d{4}))?\b', city_state_part)
            if zip_match:
                components.zip_code = zip_match.group(1)
                if zip_match.group(2):
                    components.zip_plus_four = zip_match.group(2)
                
                # Remove zip from city/state part
                city_state_part = city_state_part[:zip_match.start()].strip()
            
            # Extract state
            state_match = re.search(r'\b([A-Z]{2})\b', city_state_part)
            if state_match:
                components.state = state_match.group(1)
                
                # Remove state from city part
                city_part = city_state_part[:state_match.start()].strip()
                if city_part:
                    components.city = city_part.rstrip(',')
        
        return components

# Factory function
def create_openaddresses_client(api_key: Optional[str] = None,
                              cache_dir: Optional[Path] = None) -> OpenAddressesClient:
    """Create configured OpenAddresses client."""
    return OpenAddressesClient(
        api_key=api_key,
        cache_dir=cache_dir
    )

if __name__ == "__main__":
    # Test the OpenAddresses client
    client = create_openaddresses_client()
    
    # Test address search
    test_address = "123 Main St, Madison, WI"
    print(f"Testing address search: {test_address}")
    
    # Note: This would require async context in real usage
    # result = await client.search_address(test_address)
    # print(f"Result: {result.to_dict()}")
    
    print("OpenAddresses client initialized successfully")