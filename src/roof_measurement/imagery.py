"""
Aerial imagery provider integration
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import base64
import io
from PIL import Image
import numpy as np


class AerialImageryProvider:
    """Provider for aerial imagery from multiple sources"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize imagery provider
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # API keys and endpoints
        self.openaerialmap_key = config.get('openaerialmap_api_key')
        self.mapbox_key = config.get('mapbox_api_key')
        self.google_maps_key = config.get('google_maps_api_key')
        self.nearmap_key = config.get('nearmap_api_key')
        
        # Base URLs
        self.openaerialmap_url = "https://api.openaerialmap.org"
        self.mapbox_url = "https://api.mapbox.com"
        self.google_maps_url = "https://maps.googleapis.com/maps/api"
        self.nearmap_url = "https://api.nearmap.com"
        
        # Cache for imagery requests
        self.imagery_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour
        
        # Rate limiting
        self.rate_limits = {
            'requests_per_minute': config.get('rate_limit', 60),
            'requests_per_hour': config.get('hourly_rate_limit', 1000)
        }
        self.request_timestamps = []
        
        self.logger.info("AerialImageryProvider initialized")
    
    async def get_imagery(
        self,
        latitude: float,
        longitude: float,
        zoom_level: int = 20,
        image_size: Tuple[int, int] = (1024, 1024),
        sources: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get aerial imagery for a location
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            zoom_level: Zoom level (1-20)
            image_size: Image dimensions (width, height)
            sources: List of preferred sources
            date_range: Optional date range for imagery
            
        Returns:
            Dict containing imagery data and metadata
        """
        if sources is None:
            sources = ['openaerialmap', 'mapbox', 'google', 'nearmap']
        
        # Check cache first
        cache_key = f"{latitude}_{longitude}_{zoom_level}_{image_size[0]}_{image_size[1]}"
        cached_result = self._check_cache(cache_key)
        if cached_result:
            self.logger.info(f"Returning cached imagery for {cache_key}")
            return cached_result
        
        # Try each source in order
        for source in sources:
            try:
                imagery_data = await self._get_imagery_from_source(
                    source, latitude, longitude, zoom_level, image_size, date_range
                )
                if imagery_data:
                    # Cache the result
                    self._cache_result(cache_key, imagery_data)
                    return imagery_data
            except Exception as e:
                self.logger.warning(f"Failed to get imagery from {source}: {str(e)}")
                continue
        
        # If all sources fail, raise error
        raise Exception("Failed to get imagery from any source")
    
    async def _get_imagery_from_source(
        self,
        source: str,
        latitude: float,
        longitude: float,
        zoom_level: int,
        image_size: Tuple[int, int],
        date_range: Optional[Tuple[datetime, datetime]]
    ) -> Optional[Dict[str, Any]]:
        """Get imagery from specific source"""
        
        # Check rate limiting
        if not self._check_rate_limit():
            await asyncio.sleep(1)
        
        if source == 'openaerialmap':
            return await self._get_openaerialmap_imagery(
                latitude, longitude, zoom_level, image_size, date_range
            )
        elif source == 'mapbox':
            return await self._get_mapbox_imagery(
                latitude, longitude, zoom_level, image_size
            )
        elif source == 'google':
            return await self._get_google_imagery(
                latitude, longitude, zoom_level, image_size
            )
        elif source == 'nearmap':
            return await self._get_nearmap_imagery(
                latitude, longitude, zoom_level, image_size, date_range
            )
        else:
            raise ValueError(f"Unknown imagery source: {source}")
    
    async def _get_openaerialmap_imagery(
        self,
        latitude: float,
        longitude: float,
        zoom_level: int,
        image_size: Tuple[int, int],
        date_range: Optional[Tuple[datetime, datetime]]
    ) -> Optional[Dict[str, Any]]:
        """Get imagery from OpenAerialMap"""
        
        try:
            # Search for available imagery
            search_params = {
                'bbox': self._get_bbox(latitude, longitude, 0.001),  # Small bbox
                'limit': 10
            }
            
            if date_range:
                search_params['acquired_from'] = date_range[0].isoformat()
                search_params['acquired_to'] = date_range[1].isoformat()
            
            async with aiohttp.ClientSession() as session:
                # Search for imagery
                search_url = f"{self.openaerialmap_url}/meta"
                async with session.get(search_url, params=search_params) as response:
                    if response.status == 200:
                        search_results = await response.json()
                        
                        if search_results.get('results'):
                            # Get the most recent image
                            image_info = search_results['results'][0]
                            image_id = image_info['uuid']
                            
                            # Get the actual image
                            image_url = f"{self.openaerialmap.org}/images/{image_id}"
                            async with session.get(image_url) as img_response:
                                if img_response.status == 200:
                                    image_data = await img_response.read()
                                    
                                    return {
                                        'source': 'openaerialmap',
                                        'image_data': image_data,
                                        'metadata': {
                                            'image_id': image_id,
                                            'acquisition_date': image_info.get('acquisition_end'),
                                            'resolution': image_info.get('gsd'),
                                            'provider': image_info.get('provider'),
                                            'platform': image_info.get('platform'),
                                            'coordinates': (latitude, longitude),
                                            'zoom_level': zoom_level,
                                            'image_size': image_size
                                        }
                                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting OpenAerialMap imagery: {str(e)}")
            return None
    
    async def _get_mapbox_imagery(
        self,
        latitude: float,
        longitude: float,
        zoom_level: int,
        image_size: Tuple[int, int]
    ) -> Optional[Dict[str, Any]]:
        """Get satellite imagery from Mapbox"""
        
        if not self.mapbox_key:
            self.logger.warning("Mapbox API key not configured")
            return None
        
        try:
            # Mapbox Static Images API
            # Convert lat/lon to tile coordinates
            x, y = self._lat_lon_to_tile_x_y(latitude, longitude, zoom_level)
            
            # Construct URL
            url = (
                f"{self.mapbox_url}/styles/v1/mapbox/satellite-v9/static/"
                f"{x},{y},{zoom_level}/{image_size[0]}x{image_size[1]}@2x"
                f"?access_token={self.mapbox_key}"
            )
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        
                        return {
                            'source': 'mapbox',
                            'image_data': image_data,
                            'metadata': {
                                'coordinates': (latitude, longitude),
                                'zoom_level': zoom_level,
                                'image_size': image_size,
                                'tile_coords': (x, y),
                                'style': 'satellite-v9',
                                'resolution': 'high'
                            }
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting Mapbox imagery: {str(e)}")
            return None
    
    async def _get_google_imagery(
        self,
        latitude: float,
        longitude: float,
        zoom_level: int,
        image_size: Tuple[int, int]
    ) -> Optional[Dict[str, Any]]:
        """Get satellite imagery from Google Maps"""
        
        if not self.google_maps_key:
            self.logger.warning("Google Maps API key not configured")
            return None
        
        try:
            # Google Static Maps API
            params = {
                'center': f"{latitude},{longitude}",
                'zoom': zoom_level,
                'size': f"{image_size[0]}x{image_size[1]}",
                'maptype': 'satellite',
                'key': self.google_maps_key
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.google_maps_url}/staticmap"
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        
                        return {
                            'source': 'google',
                            'image_data': image_data,
                            'metadata': {
                                'coordinates': (latitude, longitude),
                                'zoom_level': zoom_level,
                                'image_size': image_size,
                                'maptype': 'satellite'
                            }
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting Google Maps imagery: {str(e)}")
            return None
    
    async def _get_nearmap_imagery(
        self,
        latitude: float,
        longitude: float,
        zoom_level: int,
        image_size: Tuple[int, int],
        date_range: Optional[Tuple[datetime, datetime]]
    ) -> Optional[Dict[str, Any]]:
        """Get high-resolution imagery from Nearmap"""
        
        if not self.nearmap_key:
            self.logger.warning("Nearmap API key not configured")
            return None
        
        try:
            # Nearmap Static Image API
            params = {
                'center': f"{latitude},{longitude}",
                'zoom': zoom_level,
                'size': f"{image_size[0]}x{image_size[1]}",
                'apikey': self.nearmap_key,
                'httpauth': 'false'
            }
            
            if date_range:
                params['date'] = date_range[1].strftime('%Y-%m-%d')
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.nearmap_url}/staticmap"
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        
                        return {
                            'source': 'nearmap',
                            'image_data': image_data,
                            'metadata': {
                                'coordinates': (latitude, longitude),
                                'zoom_level': zoom_level,
                                'image_size': image_size,
                                'date_range': date_range,
                                'resolution': 'ultra-high'
                            }
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting Nearmap imagery: {str(e)}")
            return None
    
    def _get_bbox(self, latitude: float, longitude: float, radius: float) -> str:
        """Get bounding box for search"""
        lat_min = latitude - radius
        lat_max = latitude + radius
        lon_min = longitude - radius
        lon_max = longitude + radius
        return f"{lon_min},{lat_min},{lon_max},{lat_max}"
    
    def _lat_lon_to_tile_x_y(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates"""
        import math
        
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        
        return x, y
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if imagery is cached and still valid"""
        if cache_key in self.imagery_cache:
            cached_data = self.imagery_cache[cache_key]
            if datetime.now() - cached_data['cached_at'] < timedelta(seconds=self.cache_ttl):
                return cached_data['data']
            else:
                # Remove expired cache entry
                del self.imagery_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, data: Dict[str, Any]):
        """Cache imagery result"""
        self.imagery_cache[cache_key] = {
            'data': data,
            'cached_at': datetime.now()
        }
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows request"""
        now = datetime.now()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if now - ts < timedelta(minutes=1)
        ]
        
        # Check if we're under the rate limit
        if len(self.request_timestamps) < self.rate_limits['requests_per_minute']:
            self.request_timestamps.append(now)
            return True
        else:
            return False
    
    async def get_imagery_metadata(
        self,
        latitude: float,
        longitude: float,
        radius: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Get metadata about available imagery for a location
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            radius: Search radius in degrees
            
        Returns:
            List of available imagery metadata
        """
        bbox = self._get_bbox(latitude, longitude, radius)
        
        metadata_list = []
        
        # Check OpenAerialMap
        if self.openaerialmap_key:
            try:
                async with aiohttp.ClientSession() as session:
                    search_url = f"{self.openaerialmap_url}/meta"
                    params = {'bbox': bbox, 'limit': 20}
                    
                    async with session.get(search_url, params=params) as response:
                        if response.status == 200:
                            results = await response.json()
                            for result in results.get('results', []):
                                metadata_list.append({
                                    'source': 'openaerialmap',
                                    'image_id': result['uuid'],
                                    'acquisition_date': result.get('acquisition_end'),
                                    'resolution': result.get('gsd'),
                                    'provider': result.get('provider'),
                                    'platform': result.get('platform')
                                })
            except Exception as e:
                self.logger.error(f"Error getting OpenAerialMap metadata: {str(e)}")
        
        return metadata_list
    
    def clear_cache(self):
        """Clear imagery cache"""
        self.imagery_cache.clear()
        self.logger.info("Imagery cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.imagery_cache),
            'cache_ttl': self.cache_ttl,
            'oldest_cache': min(
                (data['cached_at'] for data in self.imagery_cache.values()),
                default=None
            ),
            'newest_cache': max(
                (data['cached_at'] for data in self.imagery_cache.values()),
                default=None
            )
        }