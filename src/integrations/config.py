"""
Integration configuration management for Hail Hero CRM.

This module handles configuration for GoHighLevel and Twilio integrations,
including API credentials, webhook endpoints, and field mapping.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import json
from enum import Enum

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Integration types supported by the system."""
    GOHIGHLEVEL = "gohighlevel"
    TWILIO = "twilio"


class WebhookEvent(Enum):
    """Webhook event types."""
    CONTACT_CREATED = "contact_created"
    CONTACT_UPDATED = "contact_updated"
    SMS_RECEIVED = "sms_received"
    SMS_SENT = "sms_sent"
    CALL_STARTED = "call_started"
    CALL_ENDED = "call_ended"


@dataclass
class GoHighLevelConfig:
    """Configuration for GoHighLevel integration."""
    # API Configuration
    api_key: Optional[str] = None
    base_url: str = "https://api.gohighlevel.com/v1"
    
    # Webhook Configuration
    webhook_secret: Optional[str] = None
    webhook_url: Optional[str] = None
    
    # Field Mapping
    field_mapping: Dict[str, str] = field(default_factory=lambda: {
        'first_name': 'firstName',
        'last_name': 'lastName',
        'phone': 'phone',
        'email': 'email',
        'address': 'address',
        'city': 'city',
        'state': 'state',
        'zip_code': 'postalCode',
        'lead_source': 'source',
        'lead_score': 'customField.lead_score',
        'tags': 'tags',
        'notes': 'notes'
    })
    
    # Request Configuration
    timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 0.1
    
    # Data Processing
    batch_size: int = 100
    enable_webhooks: bool = True
    sync_contact_updates: bool = True
    
    # Location ID (required for GoHighLevel)
    location_id: Optional[str] = None


@dataclass
class TwilioConfig:
    """Configuration for Twilio integration."""
    # API Configuration
    account_sid: Optional[str] = None
    auth_token: Optional[str] = None
    phone_number: Optional[str] = None
    
    # Webhook Configuration
    webhook_secret: Optional[str] = None
    webhook_url: Optional[str] = None
    
    # Messaging Configuration
    messaging_service_sid: Optional[str] = None
    enable_mms: bool = True
    max_message_length: int = 1600
    
    # Request Configuration
    timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 0.1
    
    # Consent Management
    require_consent: bool = True
    consent_text: str = "Reply STOP to unsubscribe. Msg&data rates may apply."
    default_country_code: str = "+1"


@dataclass
class IntegrationConfig:
    """Main integration configuration."""
    gohighlevel: GoHighLevelConfig = field(default_factory=GoHighLevelConfig)
    twilio: TwilioConfig = field(default_factory=TwilioConfig)
    
    # System-wide settings
    enable_integrations: bool = True
    enable_webhooks: bool = True
    debug_mode: bool = False
    
    # Security
    encryption_key: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_webhook_payloads: bool = False
    log_sensitive_data: bool = False


class IntegrationConfigManager:
    """Configuration manager for handling integration settings."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("integrations.json")
        self.config = IntegrationConfig()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables and config file."""
        # Load from config file if it exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self._update_config_from_dict(config_data)
                logger.info(f"Loaded integration configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load integration config file: {e}")
        
        # Override with environment variables
        self._load_from_environment()
        
        # Validate configuration
        self._validate_config()
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        # GoHighLevel Configuration
        if 'gohighlevel' in config_data:
            ghl_data = config_data['gohighlevel']
            self.config.gohighlevel.api_key = ghl_data.get('api_key', self.config.gohighlevel.api_key)
            self.config.gohighlevel.base_url = ghl_data.get('base_url', self.config.gohighlevel.base_url)
            self.config.gohighlevel.webhook_secret = ghl_data.get('webhook_secret', self.config.gohighlevel.webhook_secret)
            self.config.gohighlevel.webhook_url = ghl_data.get('webhook_url', self.config.gohighlevel.webhook_url)
            self.config.gohighlevel.location_id = ghl_data.get('location_id', self.config.gohighlevel.location_id)
            self.config.gohighlevel.field_mapping.update(ghl_data.get('field_mapping', {}))
            self.config.gohighlevel.timeout = ghl_data.get('timeout', self.config.gohighlevel.timeout)
            self.config.gohighlevel.max_retries = ghl_data.get('max_retries', self.config.gohighlevel.max_retries)
            self.config.gohighlevel.enable_webhooks = ghl_data.get('enable_webhooks', self.config.gohighlevel.enable_webhooks)
            self.config.gohighlevel.sync_contact_updates = ghl_data.get('sync_contact_updates', self.config.gohighlevel.sync_contact_updates)
        
        # Twilio Configuration
        if 'twilio' in config_data:
            twilio_data = config_data['twilio']
            self.config.twilio.account_sid = twilio_data.get('account_sid', self.config.twilio.account_sid)
            self.config.twilio.auth_token = twilio_data.get('auth_token', self.config.twilio.auth_token)
            self.config.twilio.phone_number = twilio_data.get('phone_number', self.config.twilio.phone_number)
            self.config.twilio.webhook_secret = twilio_data.get('webhook_secret', self.config.twilio.webhook_secret)
            self.config.twilio.webhook_url = twilio_data.get('webhook_url', self.config.twilio.webhook_url)
            self.config.twilio.messaging_service_sid = twilio_data.get('messaging_service_sid', self.config.twilio.messaging_service_sid)
            self.config.twilio.enable_mms = twilio_data.get('enable_mms', self.config.twilio.enable_mms)
            self.config.twilio.require_consent = twilio_data.get('require_consent', self.config.twilio.require_consent)
            self.config.twilio.consent_text = twilio_data.get('consent_text', self.config.twilio.consent_text)
            self.config.twilio.timeout = twilio_data.get('timeout', self.config.twilio.timeout)
            self.config.twilio.max_retries = twilio_data.get('max_retries', self.config.twilio.max_retries)
        
        # System Configuration
        if 'system' in config_data:
            sys_data = config_data['system']
            self.config.enable_integrations = sys_data.get('enable_integrations', self.config.enable_integrations)
            self.config.enable_webhooks = sys_data.get('enable_webhooks', self.config.enable_webhooks)
            self.config.debug_mode = sys_data.get('debug_mode', self.config.debug_mode)
            self.config.log_level = sys_data.get('log_level', self.config.log_level)
            self.config.log_webhook_payloads = sys_data.get('log_webhook_payloads', self.config.log_webhook_payloads)
            self.config.log_sensitive_data = sys_data.get('log_sensitive_data', self.config.log_sensitive_data)
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # GoHighLevel Configuration
        self.config.gohighlevel.api_key = os.getenv('GOHIGHLEVEL_API_KEY', self.config.gohighlevel.api_key)
        self.config.gohighlevel.base_url = os.getenv('GOHIGHLEVEL_BASE_URL', self.config.gohighlevel.base_url)
        self.config.gohighlevel.webhook_secret = os.getenv('GOHIGHLEVEL_WEBHOOK_SECRET', self.config.gohighlevel.webhook_secret)
        self.config.gohighlevel.webhook_url = os.getenv('GOHIGHLEVEL_WEBHOOK_URL', self.config.gohighlevel.webhook_url)
        self.config.gohighlevel.location_id = os.getenv('GOHIGHLEVEL_LOCATION_ID', self.config.gohighlevel.location_id)
        self.config.gohighlevel.timeout = int(os.getenv('GOHIGHLEVEL_TIMEOUT', str(self.config.gohighlevel.timeout)))
        self.config.gohighlevel.max_retries = int(os.getenv('GOHIGHLEVEL_MAX_RETRIES', str(self.config.gohighlevel.max_retries)))
        self.config.gohighlevel.enable_webhooks = os.getenv('GOHIGHLEVEL_ENABLE_WEBHOOKS', 'true').lower() == 'true'
        
        # Twilio Configuration
        self.config.twilio.account_sid = os.getenv('TWILIO_ACCOUNT_SID', self.config.twilio.account_sid)
        self.config.twilio.auth_token = os.getenv('TWILIO_AUTH_TOKEN', self.config.twilio.auth_token)
        self.config.twilio.phone_number = os.getenv('TWILIO_PHONE_NUMBER', self.config.twilio.phone_number)
        self.config.twilio.webhook_secret = os.getenv('TWILIO_WEBHOOK_SECRET', self.config.twilio.webhook_secret)
        self.config.twilio.webhook_url = os.getenv('TWILIO_WEBHOOK_URL', self.config.twilio.webhook_url)
        self.config.twilio.messaging_service_sid = os.getenv('TWILIO_MESSAGING_SERVICE_SID', self.config.twilio.messaging_service_sid)
        self.config.twilio.enable_mms = os.getenv('TWILIO_ENABLE_MMS', 'true').lower() == 'true'
        self.config.twilio.require_consent = os.getenv('TWILIO_REQUIRE_CONSENT', 'true').lower() == 'true'
        self.config.twilio.consent_text = os.getenv('TWILIO_CONSENT_TEXT', self.config.twilio.consent_text)
        self.config.twilio.timeout = int(os.getenv('TWILIO_TIMEOUT', str(self.config.twilio.timeout)))
        self.config.twilio.max_retries = int(os.getenv('TWILIO_MAX_RETRIES', str(self.config.twilio.max_retries)))
        
        # System Configuration
        self.config.enable_integrations = os.getenv('ENABLE_INTEGRATIONS', 'true').lower() == 'true'
        self.config.enable_webhooks = os.getenv('ENABLE_WEBHOOKS', 'true').lower() == 'true'
        self.config.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        self.config.log_level = os.getenv('INTEGRATION_LOG_LEVEL', self.config.log_level)
        self.config.log_webhook_payloads = os.getenv('LOG_WEBHOOK_PAYLOADS', 'false').lower() == 'true'
        self.config.log_sensitive_data = os.getenv('LOG_SENSITIVE_DATA', 'false').lower() == 'true'
        self.config.encryption_key = os.getenv('ENCRYPTION_KEY', self.config.encryption_key)
    
    def _validate_config(self):
        """Validate configuration settings."""
        errors = []
        
        # Validate GoHighLevel configuration
        if self.config.enable_integrations:
            if not self.config.gohighlevel.api_key:
                errors.append("GOHIGHLEVEL_API_KEY is required when integrations are enabled")
            
            if not self.config.gohighlevel.location_id:
                errors.append("GOHIGHLEVEL_LOCATION_ID is required for GoHighLevel integration")
            
            if self.config.gohighlevel.timeout <= 0:
                errors.append("GoHighLevel timeout must be positive")
            
            if self.config.gohighlevel.max_retries < 0:
                errors.append("GoHighLevel max_retries must be non-negative")
        
        # Validate Twilio configuration
        if self.config.enable_integrations:
            if not self.config.twilio.account_sid:
                errors.append("TWILIO_ACCOUNT_SID is required when integrations are enabled")
            
            if not self.config.twilio.auth_token:
                errors.append("TWILIO_AUTH_TOKEN is required when integrations are enabled")
            
            if not self.config.twilio.phone_number:
                errors.append("TWILIO_PHONE_NUMBER is required when integrations are enabled")
            
            if self.config.twilio.timeout <= 0:
                errors.append("Twilio timeout must be positive")
            
            if self.config.twilio.max_retries < 0:
                errors.append("Twilio max_retries must be non-negative")
        
        if errors:
            raise ValueError(f"Integration configuration validation failed: {'; '.join(errors)}")
        
        logger.info("Integration configuration validation passed")
    
    def get_gohighlevel_config(self) -> GoHighLevelConfig:
        """Get GoHighLevel configuration."""
        return self.config.gohighlevel
    
    def get_twilio_config(self) -> TwilioConfig:
        """Get Twilio configuration."""
        return self.config.twilio
    
    def get_integration_config(self) -> IntegrationConfig:
        """Get main integration configuration."""
        return self.config
    
    def save_config(self):
        """Save current configuration to file."""
        config_data = {
            'gohighlevel': {
                'api_key': self.config.gohighlevel.api_key,
                'base_url': self.config.gohighlevel.base_url,
                'webhook_secret': self.config.gohighlevel.webhook_secret,
                'webhook_url': self.config.gohighlevel.webhook_url,
                'location_id': self.config.gohighlevel.location_id,
                'field_mapping': self.config.gohighlevel.field_mapping,
                'timeout': self.config.gohighlevel.timeout,
                'max_retries': self.config.gohighlevel.max_retries,
                'enable_webhooks': self.config.gohighlevel.enable_webhooks,
                'sync_contact_updates': self.config.gohighlevel.sync_contact_updates,
            },
            'twilio': {
                'account_sid': self.config.twilio.account_sid,
                'auth_token': self.config.twilio.auth_token,
                'phone_number': self.config.twilio.phone_number,
                'webhook_secret': self.config.twilio.webhook_secret,
                'webhook_url': self.config.twilio.webhook_url,
                'messaging_service_sid': self.config.twilio.messaging_service_sid,
                'enable_mms': self.config.twilio.enable_mms,
                'require_consent': self.config.twilio.require_consent,
                'consent_text': self.config.twilio.consent_text,
                'timeout': self.config.twilio.timeout,
                'max_retries': self.config.twilio.max_retries,
            },
            'system': {
                'enable_integrations': self.config.enable_integrations,
                'enable_webhooks': self.config.enable_webhooks,
                'debug_mode': self.config.debug_mode,
                'log_level': self.config.log_level,
                'log_webhook_payloads': self.config.log_webhook_payloads,
                'log_sensitive_data': self.config.log_sensitive_data,
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Integration configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save integration configuration: {e}")
            raise
    
    def is_enabled(self, integration_type: IntegrationType) -> bool:
        """Check if an integration is enabled."""
        if not self.config.enable_integrations:
            return False
        
        if integration_type == IntegrationType.GOHIGHLEVEL:
            return bool(self.config.gohighlevel.api_key and self.config.gohighlevel.location_id)
        elif integration_type == IntegrationType.TWILIO:
            return bool(self.config.twilio.account_sid and self.config.twilio.auth_token and self.config.twilio.phone_number)
        
        return False
    
    def get_webhook_url(self, integration_type: IntegrationType) -> Optional[str]:
        """Get webhook URL for an integration."""
        if integration_type == IntegrationType.GOHIGHLEVEL:
            return self.config.gohighlevel.webhook_url
        elif integration_type == IntegrationType.TWILIO:
            return self.config.twilio.webhook_url
        return None
    
    def get_webhook_secret(self, integration_type: IntegrationType) -> Optional[str]:
        """Get webhook secret for an integration."""
        if integration_type == IntegrationType.GOHIGHLEVEL:
            return self.config.gohighlevel.webhook_secret
        elif integration_type == IntegrationType.TWILIO:
            return self.config.twilio.webhook_secret
        return None


def get_integration_config() -> IntegrationConfig:
    """Get global integration configuration instance."""
    if not hasattr(get_integration_config, '_instance'):
        get_integration_config._instance = IntegrationConfigManager()
    return get_integration_config._instance.get_integration_config()


def get_gohighlevel_config() -> GoHighLevelConfig:
    """Get GoHighLevel configuration."""
    return get_integration_config().gohighlevel


def get_twilio_config() -> TwilioConfig:
    """Get Twilio configuration."""
    return get_integration_config().twilio