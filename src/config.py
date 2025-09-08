"""
Configuration management for Hail Hero NOAA integration.

This module handles environment variables, API configuration, and system settings
for production-ready NOAA data integration.
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class NOAAConfig:
    """Configuration for NOAA/NCEI API integration."""
    # API Configuration
    ncei_token: Optional[str] = None
    ncei_base_url: str = "https://www.ncei.noaa.gov/access/services/search/v1/data"
    ncei_dataset: str = "stormevents"
    
    # Request Configuration
    timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    retry_backoff_factor: float = 2.0
    
    # Data Processing
    batch_size: int = 1000
    max_events_per_request: int = 10000
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    
    # Data Filtering
    default_hail_min: float = 0.5
    default_wind_min: float = 60.0
    target_states: list = field(default_factory=lambda: ['WISCONSIN', 'ILLINOIS'])
    
    # Geographic Constraints
    min_latitude: float = 41.5  # Northern Illinois boundary
    max_latitude: float = 47.0  # Northern Wisconsin boundary
    min_longitude: float = -93.0  # Western Wisconsin boundary
    max_longitude: float = -87.0  # Eastern Illinois boundary
    
    # Property Enrichment
    enable_property_enrichment: bool = True
    max_properties_per_event: int = 50
    search_radius_miles: float = 5.0
    property_quality_threshold: float = 60.0
    
    # Logging and Monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 8080
    
    # Scheduling
    enable_scheduling: bool = True
    fetch_interval_hours: int = 6
    enable_incremental_fetch: bool = True
    
    # Data Storage
    data_directory: str = "data"
    cache_directory: str = "cache"
    backup_enabled: bool = True
    backup_retention_days: int = 30


@dataclass
class DatabaseConfig:
    """Database configuration for data storage."""
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 5432
    database: str = "hail_hero"
    username: Optional[str] = None
    password: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class ExternalServicesConfig:
    """Configuration for external services."""
    # Geocoding
    nominatim_base_url: str = "https://nominatim.openstreetmap.org"
    nominatim_timeout: int = 10
    nominatim_rate_limit: float = 1.0
    
    # Property Data
    openaddresses_enabled: bool = True
    openaddresses_base_url: str = "https://openaddresses.io"
    
    # Monitoring
    enable_health_checks: bool = True
    health_check_interval_minutes: int = 5


@dataclass
class SystemConfig:
    """Main system configuration."""
    noaa: NOAAConfig = field(default_factory=NOAAConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    external_services: ExternalServicesConfig = field(default_factory=ExternalServicesConfig)
    
    # System-wide settings
    debug: bool = False
    environment: str = "development"
    version: str = "1.0.0"
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    log_file: Path = field(default_factory=lambda: Path("logs/noaa_integration.log"))
    
    # Security
    secret_key: Optional[str] = None
    enable_cors: bool = True
    
    # Performance
    max_workers: int = 4
    enable_async_processing: bool = True


class ConfigManager:
    """Configuration manager for handling environment variables and settings."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("config.json")
        self.config = SystemConfig()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables and config file."""
        # Load from config file if it exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self._update_config_from_dict(config_data)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # Override with environment variables
        self._load_from_environment()
        
        # Set up paths
        self._setup_paths()
        
        # Validate configuration
        self._validate_config()
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        # NOAA Configuration
        if 'noaa' in config_data:
            noaa_data = config_data['noaa']
            self.config.noaa.ncei_token = noaa_data.get('ncei_token', self.config.noaa.ncei_token)
            self.config.noaa.ncei_base_url = noaa_data.get('ncei_base_url', self.config.noaa.ncei_base_url)
            self.config.noaa.timeout = noaa_data.get('timeout', self.config.noaa.timeout)
            self.config.noaa.max_retries = noaa_data.get('max_retries', self.config.noaa.max_retries)
            self.config.noaa.rate_limit_delay = noaa_data.get('rate_limit_delay', self.config.noaa.rate_limit_delay)
            self.config.noaa.enable_caching = noaa_data.get('enable_caching', self.config.noaa.enable_caching)
            self.config.noaa.enable_property_enrichment = noaa_data.get('enable_property_enrichment', self.config.noaa.enable_property_enrichment)
            self.config.noaa.enable_scheduling = noaa_data.get('enable_scheduling', self.config.noaa.enable_scheduling)
            self.config.noaa.fetch_interval_hours = noaa_data.get('fetch_interval_hours', self.config.noaa.fetch_interval_hours)
        
        # Database Configuration
        if 'database' in config_data:
            db_data = config_data['database']
            self.config.database.url = db_data.get('url', self.config.database.url)
            self.config.database.host = db_data.get('host', self.config.database.host)
            self.config.database.port = db_data.get('port', self.config.database.port)
            self.config.database.database = db_data.get('database', self.config.database.database)
            self.config.database.username = db_data.get('username', self.config.database.username)
            self.config.database.password = db_data.get('password', self.config.database.password)
        
        # External Services
        if 'external_services' in config_data:
            ext_data = config_data['external_services']
            self.config.external_services.nominatim_base_url = ext_data.get('nominatim_base_url', self.config.external_services.nominatim_base_url)
            self.config.external_services.openaddresses_enabled = ext_data.get('openaddresses_enabled', self.config.external_services.openaddresses_enabled)
        
        # System Configuration
        if 'system' in config_data:
            sys_data = config_data['system']
            self.config.debug = sys_data.get('debug', self.config.debug)
            self.config.environment = sys_data.get('environment', self.config.environment)
            self.config.log_level = sys_data.get('log_level', self.config.log_level)
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # NOAA API Configuration
        self.config.noaa.ncei_token = os.getenv('NCEI_TOKEN', self.config.noaa.ncei_token)
        self.config.noaa.ncei_base_url = os.getenv('NCEI_BASE_URL', self.config.noaa.ncei_base_url)
        self.config.noaa.timeout = int(os.getenv('NCEI_TIMEOUT', str(self.config.noaa.timeout)))
        self.config.noaa.max_retries = int(os.getenv('NCEI_MAX_RETRIES', str(self.config.noaa.max_retries)))
        self.config.noaa.rate_limit_delay = float(os.getenv('NCEI_RATE_LIMIT_DELAY', str(self.config.noaa.rate_limit_delay)))
        self.config.noaa.enable_caching = os.getenv('NCEI_ENABLE_CACHING', 'true').lower() == 'true'
        
        # Data Processing
        self.config.noaa.batch_size = int(os.getenv('NCEI_BATCH_SIZE', str(self.config.noaa.batch_size)))
        self.config.noaa.default_hail_min = float(os.getenv('HAIL_MIN_SIZE', str(self.config.noaa.default_hail_min)))
        self.config.noaa.default_wind_min = float(os.getenv('WIND_MIN_SPEED', str(self.config.noaa.default_wind_min)))
        
        # Property Enrichment
        self.config.noaa.enable_property_enrichment = os.getenv('ENABLE_PROPERTY_ENRICHMENT', 'true').lower() == 'true'
        self.config.noaa.max_properties_per_event = int(os.getenv('MAX_PROPERTIES_PER_EVENT', str(self.config.noaa.max_properties_per_event)))
        
        # Database Configuration
        self.config.database.url = os.getenv('DATABASE_URL', self.config.database.url)
        self.config.database.host = os.getenv('DB_HOST', self.config.database.host)
        self.config.database.port = int(os.getenv('DB_PORT', str(self.config.database.port)))
        self.config.database.database = os.getenv('DB_NAME', self.config.database.database)
        self.config.database.username = os.getenv('DB_USER', self.config.database.username)
        self.config.database.password = os.getenv('DB_PASSWORD', self.config.database.password)
        
        # System Configuration
        self.config.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.config.environment = os.getenv('ENVIRONMENT', self.config.environment)
        self.config.log_level = os.getenv('LOG_LEVEL', self.config.log_level)
        self.config.secret_key = os.getenv('SECRET_KEY', self.config.secret_key)
        
        # External Services
        self.config.external_services.nominatim_base_url = os.getenv('NOMINATIM_BASE_URL', self.config.external_services.nominatim_base_url)
        self.config.external_services.openaddresses_enabled = os.getenv('ENABLE_OPENADDRESSES', 'true').lower() == 'true'
    
    def _setup_paths(self):
        """Set up file paths."""
        # Set project root
        self.config.project_root = Path(__file__).resolve().parents[2]
        
        # Set up data directory
        data_dir = self.config.project_root / self.config.noaa.data_directory
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up cache directory
        cache_dir = data_dir / self.config.noaa.cache_directory
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up log directory
        log_dir = self.config.project_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Update log file path
        self.config.log_file = log_dir / "noaa_integration.log"
    
    def _validate_config(self):
        """Validate configuration settings."""
        errors = []
        
        # Validate NOAA configuration
        if not self.config.noaa.ncei_token and self.config.environment == 'production':
            errors.append("NCEI_TOKEN is required in production environment")
        
        if self.config.noaa.timeout <= 0:
            errors.append("NCEI timeout must be positive")
        
        if self.config.noaa.max_retries < 0:
            errors.append("NCEI max_retries must be non-negative")
        
        if self.config.noaa.rate_limit_delay < 0:
            errors.append("NCEI rate_limit_delay must be non-negative")
        
        # Validate geographic constraints
        if not (-90 <= self.config.noaa.min_latitude <= 90):
            errors.append("Invalid min_latitude")
        
        if not (-90 <= self.config.noaa.max_latitude <= 90):
            errors.append("Invalid max_latitude")
        
        if not (-180 <= self.config.noaa.min_longitude <= 180):
            errors.append("Invalid min_longitude")
        
        if not (-180 <= self.config.noaa.max_longitude <= 180):
            errors.append("Invalid max_longitude")
        
        # Validate database configuration
        if self.config.environment == 'production':
            if not self.config.database.url and not all([self.config.database.host, self.config.database.database]):
                errors.append("Database configuration is required in production")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        logger.info("Configuration validation passed")
    
    def get_noaa_config(self) -> NOAAConfig:
        """Get NOAA configuration."""
        return self.config.noaa
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.config.database
    
    def get_external_services_config(self) -> ExternalServicesConfig:
        """Get external services configuration."""
        return self.config.external_services
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        return self.config
    
    def save_config(self):
        """Save current configuration to file."""
        config_data = {
            'noaa': {
                'ncei_token': self.config.noaa.ncei_token,
                'ncei_base_url': self.config.noaa.ncei_base_url,
                'timeout': self.config.noaa.timeout,
                'max_retries': self.config.noaa.max_retries,
                'rate_limit_delay': self.config.noaa.rate_limit_delay,
                'enable_caching': self.config.noaa.enable_caching,
                'enable_property_enrichment': self.config.noaa.enable_property_enrichment,
                'enable_scheduling': self.config.noaa.enable_scheduling,
                'fetch_interval_hours': self.config.noaa.fetch_interval_hours,
            },
            'database': {
                'url': self.config.database.url,
                'host': self.config.database.host,
                'port': self.config.database.port,
                'database': self.config.database.database,
                'username': self.config.database.username,
                'password': self.config.database.password,
            },
            'external_services': {
                'nominatim_base_url': self.config.external_services.nominatim_base_url,
                'openaddresses_enabled': self.config.external_services.openaddresses_enabled,
            },
            'system': {
                'debug': self.config.debug,
                'environment': self.config.environment,
                'log_level': self.config.log_level,
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.config.environment == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.config.environment == 'development'
    
    def get_log_level(self) -> str:
        """Get log level."""
        return self.config.log_level


def get_config() -> SystemConfig:
    """Get global configuration instance."""
    if not hasattr(get_config, '_instance'):
        get_config._instance = ConfigManager()
    return get_config._instance.get_system_config()


def get_noaa_config() -> NOAAConfig:
    """Get NOAA configuration."""
    return get_config().noaa


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config().database


def get_external_services_config() -> ExternalServicesConfig:
    """Get external services configuration."""
    return get_config().external_services