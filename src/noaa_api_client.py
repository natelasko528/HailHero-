"""
Production-ready NOAA/NCEI API client with proper authentication, error handling,
and retry mechanisms.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import get_noaa_config, NOAAConfig

logger = logging.getLogger(__name__)


class APIErrorType(Enum):
    """Types of API errors."""
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    DATA_ERROR = "data_error"
    VALIDATION_ERROR = "validation_error"


@dataclass
class APIError:
    """API error details."""
    error_type: APIErrorType
    message: str
    status_code: Optional[int] = None
    retry_after: Optional[int] = None
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self):
        return f"{self.error_type.value}: {self.message}"


@dataclass
class APIResponse:
    """API response wrapper."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[APIError] = None
    metadata: Optional[Dict[str, Any]] = None
    request_duration: float = 0.0
    cached: bool = False


@dataclass
class FetchResult:
    """Result of data fetch operation."""
    total_events: int
    valid_events: int
    filtered_events: int
    processing_time: float
    api_calls: int
    cached_requests: int
    errors: List[APIError]
    data_quality_score: float


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, max_requests_per_second: float = 1.0):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()


class CircuitBreaker:
    """Circuit breaker for API calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection."""
        async with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half_open"
                    self.failure_count = 0
                else:
                    raise APIError(
                        APIErrorType.NETWORK_ERROR,
                        "Circuit breaker is open"
                    )
        
        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                if self.state == "half_open":
                    self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
            
            raise


class DataCache:
    """Cache for API responses."""
    
    def __init__(self, ttl_hours: int = 24):
        self.ttl_hours = ttl_hours
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    def _get_cache_key(self, url: str, params: Dict[str, Any]) -> str:
        """Generate cache key from URL and parameters."""
        param_str = json.dumps(params, sort_keys=True)
        return f"{url}:{param_str}"
    
    async def get(self, url: str, params: Dict[str, Any]) -> Optional[APIResponse]:
        """Get cached response."""
        cache_key = self._get_cache_key(url, params)
        
        async with self._lock:
            cached = self.cache.get(cache_key)
            if cached:
                if time.time() - cached['timestamp'] < self.ttl_hours * 3600:
                    logger.debug(f"Cache hit for {cache_key}")
                    response = APIResponse(
                        success=True,
                        data=cached['data'],
                        metadata=cached['metadata'],
                        cached=True
                    )
                    return response
                else:
                    # Remove expired cache entry
                    del self.cache[cache_key]
        
        return None
    
    async def set(self, url: str, params: Dict[str, Any], response: APIResponse):
        """Cache response."""
        if not response.success or response.cached:
            return
        
        cache_key = self._get_cache_key(url, params)
        
        async with self._lock:
            self.cache[cache_key] = {
                'data': response.data,
                'metadata': response.metadata,
                'timestamp': time.time()
            }
            logger.debug(f"Cached response for {cache_key}")
    
    async def clear(self):
        """Clear all cached responses."""
        async with self._lock:
            self.cache.clear()
            logger.info("Cache cleared")


class NOAAAPIClient:
    """Production-ready NOAA/NCEI API client."""
    
    def __init__(self, config: Optional[NOAAConfig] = None):
        self.config = config or get_noaa_config()
        self.rate_limiter = RateLimiter(1.0 / self.config.rate_limit_delay)
        self.circuit_breaker = CircuitBreaker()
        self.cache = DataCache(self.config.cache_ttl_hours)
        
        # HTTP session configuration
        self.session_timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cached_requests': 0,
            'retry_count': 0,
            'total_request_time': 0.0
        }
    
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
    
    def _build_url(self, endpoint: str = "") -> str:
        """Build API URL."""
        base_url = self.config.ncei_base_url.rstrip('/')
        return f"{base_url}/{endpoint.lstrip('/')}"
    
    def _build_params(self, start_date: str, end_date: str, 
                     limit: int = 1000, offset: int = 1,
                     additional_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build API parameters."""
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
        
        return params
    
    def _parse_api_error(self, response: aiohttp.ClientResponse, 
                        response_text: str) -> APIError:
        """Parse API error response."""
        status_code = response.status
        
        try:
            error_data = json.loads(response_text)
            message = error_data.get('message', error_data.get('error', 'Unknown error'))
            details = error_data
        except (json.JSONDecodeError, AttributeError):
            message = response_text or 'Unknown error'
            details = {'raw_response': response_text}
        
        # Determine error type
        if status_code == 401:
            error_type = APIErrorType.AUTHENTICATION_ERROR
        elif status_code == 429:
            error_type = APIErrorType.RATE_LIMIT_ERROR
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                details['retry_after'] = int(retry_after)
        elif status_code >= 500:
            error_type = APIErrorType.SERVER_ERROR
        elif status_code >= 400:
            error_type = APIErrorType.VALIDATION_ERROR
        else:
            error_type = APIErrorType.DATA_ERROR
        
        return APIError(
            error_type=error_type,
            message=message,
            status_code=status_code,
            details=details
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_request(self, url: str, params: Dict[str, Any]) -> APIResponse:
        """Make API request with retry logic."""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        # Check cache first
        if self.config.enable_caching:
            cached_response = await self.cache.get(url, params)
            if cached_response:
                self.metrics['cached_requests'] += 1
                return cached_response
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get(url, params=params) as response:
                response_text = await response.text()
                request_duration = time.time() - start_time
                
                if response.status == 200:
                    try:
                        data = json.loads(response_text)
                        
                        # Extract results from various response formats
                        if isinstance(data, dict):
                            results = data.get('results', data.get('data', []))
                            metadata = {k: v for k, v in data.items() if k not in ['results', 'data']}
                        else:
                            results = data
                            metadata = {}
                        
                        api_response = APIResponse(
                            success=True,
                            data=results,
                            metadata=metadata,
                            request_duration=request_duration
                        )
                        
                        # Cache successful response
                        if self.config.enable_caching:
                            await self.cache.set(url, params, api_response)
                        
                        self.metrics['successful_requests'] += 1
                        self.metrics['total_request_time'] += request_duration
                        
                        return api_response
                        
                    except json.JSONDecodeError as e:
                        error = APIError(
                            APIErrorType.DATA_ERROR,
                            f"Failed to parse JSON response: {e}",
                            status_code=response.status
                        )
                        return APIResponse(success=False, error=error, request_duration=request_duration)
                
                else:
                    error = self._parse_api_error(response, response_text)
                    return APIResponse(success=False, error=error, request_duration=request_duration)
        
        except asyncio.TimeoutError:
            error = APIError(
                APIErrorType.TIMEOUT_ERROR,
                f"Request timeout after {self.config.timeout} seconds"
            )
            return APIResponse(success=False, error=error, request_duration=time.time() - start_time)
        
        except aiohttp.ClientError as e:
            error = APIError(
                APIErrorType.NETWORK_ERROR,
                f"Network error: {str(e)}"
            )
            return APIResponse(success=False, error=error, request_duration=time.time() - start_time)
        
        except Exception as e:
            error = APIError(
                APIErrorType.DATA_ERROR,
                f"Unexpected error: {str(e)}"
            )
            return APIResponse(success=False, error=error, request_duration=time.time() - start_time)
    
    async def fetch_events(self, start_date: str, end_date: str,
                          limit: int = 1000, offset: int = 1,
                          additional_params: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Fetch storm events from NOAA API."""
        url = self._build_url()
        params = self._build_params(start_date, end_date, limit, offset, additional_params)
        
        logger.info(f"Fetching events: {start_date} to {end_date}, limit={limit}, offset={offset}")
        
        try:
            response = await self.circuit_breaker.call(self._make_request, url, params)
            
            if response.success:
                logger.info(f"Successfully fetched {len(response.data) or 0} events")
            else:
                logger.error(f"Failed to fetch events: {response.error}")
                self.metrics['failed_requests'] += 1
            
            return response
            
        except Exception as e:
            error = APIError(
                APIErrorType.NETWORK_ERROR,
                f"Circuit breaker or retry failed: {str(e)}"
            )
            self.metrics['failed_requests'] += 1
            return APIResponse(success=False, error=error)
    
    async def fetch_all_events(self, start_date: str, end_date: str,
                              batch_size: int = 1000,
                              max_events: Optional[int] = None) -> AsyncGenerator[APIResponse, None]:
        """Fetch all events with pagination."""
        offset = 1
        total_fetched = 0
        
        logger.info(f"Starting to fetch all events from {start_date} to {end_date}")
        
        while True:
            if max_events and total_fetched >= max_events:
                logger.info(f"Reached maximum events limit: {max_events}")
                break
            
            current_batch_size = min(batch_size, max_events - total_fetched) if max_events else batch_size
            
            response = await self.fetch_events(
                start_date, end_date, 
                limit=current_batch_size, 
                offset=offset
            )
            
            if not response.success:
                logger.error(f"Failed to fetch batch at offset {offset}: {response.error}")
                yield response
                break
            
            if not response.data or len(response.data) == 0:
                logger.info("No more events available")
                break
            
            yield response
            
            total_fetched += len(response.data)
            offset += len(response.data)
            
            logger.info(f"Fetched {len(response.data)} events (total: {total_fetched})")
            
            # If we got fewer events than requested, we're done
            if len(response.data) < current_batch_size:
                break
    
    async def validate_token(self) -> bool:
        """Validate API token with a test request."""
        if not self.config.ncei_token:
            logger.warning("No NCEI token configured")
            return False
        
        try:
            # Make a small test request
            test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            response = await self.fetch_events(test_date, test_date, limit=1)
            
            if response.success:
                logger.info("NCEI token validation successful")
                return True
            else:
                if response.error and response.error.error_type == APIErrorType.AUTHENTICATION_ERROR:
                    logger.error("NCEI token validation failed: authentication error")
                else:
                    logger.warning(f"NCEI token validation failed: {response.error}")
                return False
                
        except Exception as e:
            logger.error(f"NCEI token validation error: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get API client metrics."""
        metrics = self.metrics.copy()
        if metrics['total_requests'] > 0:
            metrics['success_rate'] = metrics['successful_requests'] / metrics['total_requests']
            metrics['average_request_time'] = metrics['total_request_time'] / metrics['total_requests']
        else:
            metrics['success_rate'] = 0.0
            metrics['average_request_time'] = 0.0
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the API client."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'base_url': self.config.ncei_base_url,
                'dataset': self.config.ncei_dataset,
                'timeout': self.config.timeout,
                'max_retries': self.config.max_retries,
                'rate_limit_delay': self.config.rate_limit_delay,
                'enable_caching': self.config.enable_caching,
            },
            'metrics': self.get_metrics(),
            'token_valid': await self.validate_token(),
            'session_active': self.session is not None and not self.session.closed,
            'circuit_breaker': {
                'state': self.circuit_breaker.state,
                'failure_count': self.circuit_breaker.failure_count,
                'last_failure_time': self.circuit_breaker.last_failure_time
            },
            'cache': {
                'entries': len(self.cache.cache),
                'ttl_hours': self.cache.ttl_hours
            }
        }
        
        # Determine overall health
        if not health_status['token_valid']:
            health_status['status'] = 'unhealthy'
            health_status['issues'] = ['Invalid API token']
        elif health_status['circuit_breaker']['state'] == 'open':
            health_status['status'] = 'degraded'
            health_status['issues'] = ['Circuit breaker is open']
        elif not health_status['session_active']:
            health_status['status'] = 'degraded'
            health_status['issues'] = ['HTTP session not active']
        
        return health_status