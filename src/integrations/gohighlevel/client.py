"""
GoHighLevel API Client for Hail Hero CRM Integration.

This module provides a comprehensive client for interacting with the GoHighLevel API,
including contact management, webhook handling, and data synchronization.
"""

import asyncio
import json
import logging
import time
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..integrations.config import get_gohighlevel_config

logger = logging.getLogger(__name__)


class GoHighLevelError(Exception):
    """Base exception for GoHighLevel API errors."""
    pass


class AuthenticationError(GoHighLevelError):
    """Authentication related errors."""
    pass


class RateLimitError(GoHighLevelError):
    """Rate limit exceeded errors."""
    pass


class ValidationError(GoHighLevelError):
    """Data validation errors."""
    pass


class ContactStatus(Enum):
    """Contact status options."""
    NEW = "new"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class PipelineStage(Enum):
    """Pipeline stage options."""
    LEAD = "lead"
    QUALIFIED = "qualified"
    PROPOSAL = "proposal"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


@dataclass
class Contact:
    """GoHighLevel contact data model."""
    id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = None
    company: Optional[str] = None
    website: Optional[str] = None
    lead_source: Optional[str] = None
    lead_score: Optional[int] = None
    tags: List[str] = None
    custom_fields: Dict[str, Any] = None
    notes: Optional[str] = None
    status: ContactStatus = ContactStatus.NEW
    pipeline_stage: Optional[PipelineStage] = None
    assigned_user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.custom_fields is None:
            self.custom_fields = {}


@dataclass
class Pipeline:
    """GoHighLevel pipeline data model."""
    id: str
    name: str
    stages: List[Dict[str, Any]]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Tag:
    """GoHighLevel tag data model."""
    id: str
    name: str
    color: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class WebhookEvent:
    """GoHighLevel webhook event data model."""
    event_type: str
    contact_id: Optional[str] = None
    contact_data: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    location_id: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass
class SyncResult:
    """Data synchronization result."""
    success: bool
    contacts_synced: int = 0
    contacts_created: int = 0
    contacts_updated: int = 0
    contacts_failed: int = 0
    errors: List[str] = None
    sync_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.sync_time is None:
            self.sync_time = datetime.now()


@dataclass
class ConflictResolution:
    """Contact conflict resolution strategy."""
    strategy: str  # "source_wins", "target_wins", "manual", "merge"
    source_field: Optional[str] = None
    target_field: Optional[str] = None
    merge_function: Optional[Callable] = None


class GoHighLevelClient:
    """GoHighLevel API client with comprehensive functionality."""

    def __init__(self, api_key: Optional[str] = None, location_id: Optional[str] = None):
        """Initialize the GoHighLevel client.
        
        Args:
            api_key: GoHighLevel API key (defaults to config)
            location_id: GoHighLevel location ID (defaults to config)
        """
        self.config = get_gohighlevel_config()
        
        self.api_key = api_key or self.config.api_key
        self.location_id = location_id or self.config.location_id
        self.base_url = self.config.base_url
        
        if not self.api_key:
            raise AuthenticationError("API key is required")
        
        if not self.location_id:
            raise AuthenticationError("Location ID is required")
        
        # Token management
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        self._token_refresh_callback = None
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff_factor if hasattr(self.config, 'retry_backoff_factor') else 1.0,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Setup async session
        self._async_session = None
        
        # Rate limiting
        self.last_request_time = 0
        self.rate_limit_delay = self.config.rate_limit_delay
        
        # Webhook verification
        self.webhook_secret = self.config.webhook_secret
        
        # Sync state tracking
        self._sync_lock = asyncio.Lock()
        self._last_sync_time = None
        
        logger.info(f"GoHighLevel client initialized for location: {self.location_id}")

    async def __aenter__(self):
        """Async context manager entry."""
        self._async_session = aiohttp.ClientSession(
            headers=self._get_headers(),
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._async_session:
            await self._async_session.close()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        # Use access token if available, otherwise fall back to API key
        auth_token = self.access_token if self.access_token else self.api_key
        
        return {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "User-Agent": "HailHero/1.0.0"
        }
    
    def _ensure_valid_token(self):
        """Ensure we have a valid access token, refreshing if necessary."""
        if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
            return True
        
        if self.refresh_token:
            return self._refresh_access_token()
        
        # If no refresh mechanism, API key authentication is sufficient
        return True
    
    def _refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token."""
        if not self.refresh_token:
            return False
        
        try:
            url = f"{self.base_url}/oauth/token"
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": self.config.client_id if hasattr(self.config, 'client_id') else None,
                "client_secret": self.config.client_secret if hasattr(self.config, 'client_secret') else None
            }
            
            response = self.session.post(url, json=data, timeout=self.config.timeout)
            result = self._handle_response(response)
            
            if result.get("access_token"):
                self.access_token = result["access_token"]
                self.refresh_token = result.get("refresh_token", self.refresh_token)
                expires_in = result.get("expires_in", 3600)
                self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)  # 60s buffer
                
                if self._token_refresh_callback:
                    self._token_refresh_callback(self.access_token, self.refresh_token)
                
                logger.info("Access token refreshed successfully")
                return True
            
        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
        
        return False
    
    def set_token_refresh_callback(self, callback: Callable[[str, str], None]):
        """Set callback for when tokens are refreshed."""
        self._token_refresh_callback = callback
    
    def authenticate_with_tokens(self, access_token: str, refresh_token: Optional[str] = None, expires_in: int = 3600):
        """Authenticate using OAuth tokens."""
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)  # 60s buffer
        logger.info("Authenticated with OAuth tokens")

    def _handle_rate_limit(self):
        """Handle rate limiting with delay."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response with error handling."""
        if response.status_code == 401:
            raise AuthenticationError("Invalid API key or authentication failed")
        elif response.status_code == 403:
            raise AuthenticationError("Access forbidden - check permissions")
        elif response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        elif response.status_code == 404:
            raise GoHighLevelError("Resource not found")
        elif response.status_code >= 500:
            raise GoHighLevelError(f"Server error: {response.status_code}")
        
        try:
            return response.json()
        except json.JSONDecodeError:
            if response.status_code >= 400:
                raise GoHighLevelError(f"HTTP {response.status_code}: {response.text}")
            return {"status": "success"}

    # Contact Management Methods
    def create_contact(self, contact: Contact) -> Contact:
        """Create a new contact in GoHighLevel.
        
        Args:
            contact: Contact object to create
            
        Returns:
            Created contact with ID
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/contacts/"
        data = self._contact_to_dict(contact)
        
        try:
            response = self.session.post(url, json=data, headers=self._get_headers(), timeout=self.config.timeout)
            result = self._handle_response(response)
            
            if result.get("contact"):
                contact_data = result["contact"]
                return self._dict_to_contact(contact_data)
            else:
                raise GoHighLevelError("Contact creation failed")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating contact: {e}")
            raise GoHighLevelError(f"Failed to create contact: {e}")

    def get_contact(self, contact_id: str) -> Contact:
        """Get a contact by ID.
        
        Args:
            contact_id: GoHighLevel contact ID
            
        Returns:
            Contact object
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/contacts/{contact_id}"
        
        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=self.config.timeout)
            result = self._handle_response(response)
            
            if result.get("contact"):
                return self._dict_to_contact(result["contact"])
            else:
                raise GoHighLevelError("Contact not found")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting contact {contact_id}: {e}")
            raise GoHighLevelError(f"Failed to get contact: {e}")

    def update_contact(self, contact_id: str, contact: Contact) -> Contact:
        """Update an existing contact.
        
        Args:
            contact_id: GoHighLevel contact ID
            contact: Updated contact data
            
        Returns:
            Updated contact
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/contacts/{contact_id}"
        data = self._contact_to_dict(contact)
        
        try:
            response = self.session.put(url, json=data, headers=self._get_headers(), timeout=self.config.timeout)
            result = self._handle_response(response)
            
            if result.get("contact"):
                return self._dict_to_contact(result["contact"])
            else:
                raise GoHighLevelError("Contact update failed")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error updating contact {contact_id}: {e}")
            raise GoHighLevelError(f"Failed to update contact: {e}")

    def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact.
        
        Args:
            contact_id: GoHighLevel contact ID
            
        Returns:
            True if successful
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/contacts/{contact_id}"
        
        try:
            response = self.session.delete(url, headers=self._get_headers(), timeout=self.config.timeout)
            self._handle_response(response)
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error deleting contact {contact_id}: {e}")
            raise GoHighLevelError(f"Failed to delete contact: {e}")

    def search_contacts(self, query: str, limit: int = 50) -> List[Contact]:
        """Search contacts by name, email, or phone.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching contacts
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/contacts/search"
        params = {
            "query": query,
            "limit": limit,
            "locationId": self.location_id
        }
        
        try:
            response = self.session.get(url, params=params, headers=self._get_headers(), timeout=self.config.timeout)
            result = self._handle_response(response)
            
            contacts = []
            if result.get("contacts"):
                for contact_data in result["contacts"]:
                    contacts.append(self._dict_to_contact(contact_data))
            
            return contacts
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching contacts: {e}")
            raise GoHighLevelError(f"Failed to search contacts: {e}")

    def get_contacts_by_tag(self, tag_name: str, limit: int = 100) -> List[Contact]:
        """Get contacts by tag.
        
        Args:
            tag_name: Tag name to filter by
            limit: Maximum number of results
            
        Returns:
            List of contacts with the specified tag
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/contacts/"
        params = {
            "locationId": self.location_id,
            "tag": tag_name,
            "limit": limit
        }
        
        try:
            response = self.session.get(url, params=params, headers=self._get_headers(), timeout=self.config.timeout)
            result = self._handle_response(response)
            
            contacts = []
            if result.get("contacts"):
                for contact_data in result["contacts"]:
                    contacts.append(self._dict_to_contact(contact_data))
            
            return contacts
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting contacts by tag: {e}")
            raise GoHighLevelError(f"Failed to get contacts by tag: {e}")

    # Pipeline Management Methods
    def get_pipelines(self) -> List[Pipeline]:
        """Get all pipelines for the location.
        
        Returns:
            List of pipelines
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/pipelines/"
        params = {"locationId": self.location_id}
        
        try:
            response = self.session.get(url, params=params, headers=self._get_headers(), timeout=self.config.timeout)
            result = self._handle_response(response)
            
            pipelines = []
            if result.get("pipelines"):
                for pipeline_data in result["pipelines"]:
                    pipelines.append(self._dict_to_pipeline(pipeline_data))
            
            return pipelines
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting pipelines: {e}")
            raise GoHighLevelError(f"Failed to get pipelines: {e}")

    def move_contact_to_stage(self, contact_id: str, pipeline_id: str, stage_id: str) -> bool:
        """Move a contact to a different pipeline stage.
        
        Args:
            contact_id: GoHighLevel contact ID
            pipeline_id: Pipeline ID
            stage_id: New stage ID
            
        Returns:
            True if successful
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/contacts/{contact_id}/pipeline"
        data = {
            "pipelineId": pipeline_id,
            "stageId": stage_id
        }
        
        try:
            response = self.session.post(url, json=data, headers=self._get_headers(), timeout=self.config.timeout)
            self._handle_response(response)
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error moving contact to stage: {e}")
            raise GoHighLevelError(f"Failed to move contact to stage: {e}")

    # Tag Management Methods
    def get_tags(self) -> List[Tag]:
        """Get all tags for the location.
        
        Returns:
            List of tags
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/tags/"
        params = {"locationId": self.location_id}
        
        try:
            response = self.session.get(url, params=params, headers=self._get_headers(), timeout=self.config.timeout)
            result = self._handle_response(response)
            
            tags = []
            if result.get("tags"):
                for tag_data in result["tags"]:
                    tags.append(self._dict_to_tag(tag_data))
            
            return tags
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting tags: {e}")
            raise GoHighLevelError(f"Failed to get tags: {e}")

    def create_tag(self, name: str, color: Optional[str] = None) -> Tag:
        """Create a new tag.
        
        Args:
            name: Tag name
            color: Tag color (hex code)
            
        Returns:
            Created tag
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/tags/"
        data = {
            "name": name,
            "locationId": self.location_id
        }
        
        if color:
            data["color"] = color
        
        try:
            response = self.session.post(url, json=data, headers=self._get_headers(), timeout=self.config.timeout)
            result = self._handle_response(response)
            
            if result.get("tag"):
                return self._dict_to_tag(result["tag"])
            else:
                raise GoHighLevelError("Tag creation failed")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating tag: {e}")
            raise GoHighLevelError(f"Failed to create tag: {e}")

    def add_tag_to_contact(self, contact_id: str, tag_name: str) -> bool:
        """Add a tag to a contact.
        
        Args:
            contact_id: GoHighLevel contact ID
            tag_name: Tag name to add
            
        Returns:
            True if successful
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/contacts/{contact_id}/tags"
        data = {
            "tagName": tag_name
        }
        
        try:
            response = self.session.post(url, json=data, headers=self._get_headers(), timeout=self.config.timeout)
            self._handle_response(response)
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error adding tag to contact: {e}")
            raise GoHighLevelError(f"Failed to add tag to contact: {e}")

    def remove_tag_from_contact(self, contact_id: str, tag_name: str) -> bool:
        """Remove a tag from a contact.
        
        Args:
            contact_id: GoHighLevel contact ID
            tag_name: Tag name to remove
            
        Returns:
            True if successful
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/contacts/{contact_id}/tags/{tag_name}"
        
        try:
            response = self.session.delete(url, headers=self._get_headers(), timeout=self.config.timeout)
            self._handle_response(response)
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error removing tag from contact: {e}")
            raise GoHighLevelError(f"Failed to remove tag from contact: {e}")

    # Webhook Management Methods
    def create_webhook(self, webhook_url: str, events: List[str]) -> str:
        """Create a webhook for event notifications.
        
        Args:
            webhook_url: URL to receive webhook events
            events: List of events to subscribe to
            
        Returns:
            Webhook ID
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/webhooks/"
        data = {
            "url": webhook_url,
            "events": events,
            "locationId": self.location_id
        }
        
        try:
            response = self.session.post(url, json=data, headers=self._get_headers(), timeout=self.config.timeout)
            result = self._handle_response(response)
            
            if result.get("webhook"):
                return result["webhook"]["id"]
            else:
                raise GoHighLevelError("Webhook creation failed")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating webhook: {e}")
            raise GoHighLevelError(f"Failed to create webhook: {e}")

    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook.
        
        Args:
            webhook_id: Webhook ID to delete
            
        Returns:
            True if successful
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}/webhooks/{webhook_id}"
        
        try:
            response = self.session.delete(url, headers=self._get_headers(), timeout=self.config.timeout)
            self._handle_response(response)
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error deleting webhook: {e}")
            raise GoHighLevelError(f"Failed to delete webhook: {e}")

    # Webhook Event Processing Methods
    def verify_webhook_signature(self, payload: str, signature: str, secret: Optional[str] = None) -> bool:
        """Verify webhook signature for security.
        
        Args:
            payload: Raw webhook payload
            signature: Signature from webhook headers
            secret: Webhook secret (defaults to config)
            
        Returns:
            True if signature is valid
        """
        webhook_secret = secret or self.webhook_secret
        if not webhook_secret:
            logger.warning("No webhook secret configured, skipping verification")
            return True
        
        expected_signature = hmac.new(
            webhook_secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, signature)
    
    def parse_webhook_event(self, payload: Dict[str, Any]) -> WebhookEvent:
        """Parse webhook payload into structured event.
        
        Args:
            payload: Raw webhook payload
            
        Returns:
            Parsed WebhookEvent object
        """
        try:
            event_type = payload.get('event', payload.get('type', 'unknown'))
            contact_id = payload.get('contactId', payload.get('contact', {}).get('id'))
            
            # Parse timestamp
            timestamp = None
            if 'timestamp' in payload:
                timestamp = datetime.fromisoformat(payload['timestamp'].replace('Z', '+00:00'))
            elif 'createdAt' in payload:
                timestamp = datetime.fromisoformat(payload['createdAt'].replace('Z', '+00:00'))
            
            return WebhookEvent(
                event_type=event_type,
                contact_id=contact_id,
                contact_data=payload.get('contact', payload.get('data', {})),
                timestamp=timestamp,
                location_id=payload.get('locationId', self.location_id),
                raw_payload=payload
            )
            
        except Exception as e:
            logger.error(f"Error parsing webhook event: {e}")
            raise ValidationError(f"Failed to parse webhook event: {e}")
    
    def process_webhook_event(self, event: WebhookEvent) -> bool:
        """Process a webhook event and take appropriate action.
        
        Args:
            event: Parsed webhook event
            
        Returns:
            True if processed successfully
        """
        try:
            logger.info(f"Processing webhook event: {event.event_type} for contact {event.contact_id}")
            
            if event.event_type == 'contact.created':
                return self._handle_contact_created(event)
            elif event.event_type == 'contact.updated':
                return self._handle_contact_updated(event)
            elif event.event_type == 'contact.deleted':
                return self._handle_contact_deleted(event)
            elif event.event_type == 'tag.added':
                return self._handle_tag_added(event)
            elif event.event_type == 'tag.removed':
                return self._handle_tag_removed(event)
            else:
                logger.warning(f"Unhandled webhook event type: {event.event_type}")
                return True
                
        except Exception as e:
            logger.error(f"Error processing webhook event {event.event_type}: {e}")
            return False
    
    def _handle_contact_created(self, event: WebhookEvent) -> bool:
        """Handle contact created webhook event."""
        if not event.contact_data:
            logger.warning("No contact data in contact.created event")
            return False
        
        try:
            # Convert to Contact object
            contact = self._dict_to_contact(event.contact_data)
            
            # Here you would integrate with HailHero's contact management
            # For now, just log the event
            logger.info(f"Contact created event processed: {contact.id} - {contact.first_name} {contact.last_name}")
            
            # Trigger any callbacks or business logic
            if hasattr(self, '_contact_created_callback'):
                self._contact_created_callback(contact)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling contact created event: {e}")
            return False
    
    def _handle_contact_updated(self, event: WebhookEvent) -> bool:
        """Handle contact updated webhook event."""
        if not event.contact_data:
            logger.warning("No contact data in contact.updated event")
            return False
        
        try:
            contact = self._dict_to_contact(event.contact_data)
            logger.info(f"Contact updated event processed: {contact.id} - {contact.first_name} {contact.last_name}")
            
            if hasattr(self, '_contact_updated_callback'):
                self._contact_updated_callback(contact)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling contact updated event: {e}")
            return False
    
    def _handle_contact_deleted(self, event: WebhookEvent) -> bool:
        """Handle contact deleted webhook event."""
        try:
            logger.info(f"Contact deleted event processed: {event.contact_id}")
            
            if hasattr(self, '_contact_deleted_callback'):
                self._contact_deleted_callback(event.contact_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling contact deleted event: {e}")
            return False
    
    def _handle_tag_added(self, event: WebhookEvent) -> bool:
        """Handle tag added webhook event."""
        try:
            tag_name = event.raw_payload.get('tagName', 'unknown') if event.raw_payload else 'unknown'
            logger.info(f"Tag added event processed: {tag_name} for contact {event.contact_id}")
            
            if hasattr(self, '_tag_added_callback'):
                self._tag_added_callback(event.contact_id, tag_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling tag added event: {e}")
            return False
    
    def _handle_tag_removed(self, event: WebhookEvent) -> bool:
        """Handle tag removed webhook event."""
        try:
            tag_name = event.raw_payload.get('tagName', 'unknown') if event.raw_payload else 'unknown'
            logger.info(f"Tag removed event processed: {tag_name} for contact {event.contact_id}")
            
            if hasattr(self, '_tag_removed_callback'):
                self._tag_removed_callback(event.contact_id, tag_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling tag removed event: {e}")
            return False
    
    def set_webhook_callbacks(self, 
                            contact_created_callback=None,
                            contact_updated_callback=None,
                            contact_deleted_callback=None,
                            tag_added_callback=None,
                            tag_removed_callback=None):
        """Set webhook event callbacks."""
        self._contact_created_callback = contact_created_callback
        self._contact_updated_callback = contact_updated_callback
        self._contact_deleted_callback = contact_deleted_callback
        self._tag_added_callback = tag_added_callback
        self._tag_removed_callback = tag_removed_callback

    # Data Transformation Methods
    def _contact_to_dict(self, contact: Contact) -> Dict[str, Any]:
        """Convert Contact object to API-compatible dictionary."""
        data = {
            "locationId": self.location_id,
        }
        
        # Add standard fields
        if contact.first_name:
            data["firstName"] = contact.first_name
        if contact.last_name:
            data["lastName"] = contact.last_name
        if contact.email:
            data["email"] = contact.email
        if contact.phone:
            data["phone"] = contact.phone
        if contact.address:
            data["address"] = contact.address
        if contact.city:
            data["city"] = contact.city
        if contact.state:
            data["state"] = contact.state
        if contact.zip_code:
            data["postalCode"] = contact.zip_code
        if contact.country:
            data["country"] = contact.country
        if contact.company:
            data["company"] = contact.company
        if contact.website:
            data["website"] = contact.website
        if contact.lead_source:
            data["source"] = contact.lead_source
        if contact.notes:
            data["notes"] = contact.notes
        if contact.assigned_user_id:
            data["assignedUser"] = contact.assigned_user_id
        
        # Add tags
        if contact.tags:
            data["tags"] = contact.tags
        
        # Add custom fields
        if contact.custom_fields:
            data["customFields"] = contact.custom_fields
        
        # Add lead score if available
        if contact.lead_score is not None:
            data["customFields"] = data.get("customFields", {})
            data["customFields"]["lead_score"] = contact.lead_score
        
        return data

    def _dict_to_contact(self, data: Dict[str, Any]) -> Contact:
        """Convert API response to Contact object."""
        # Extract custom fields
        custom_fields = data.get("customFields", {})
        lead_score = custom_fields.get("lead_score")
        
        # Parse dates
        created_at = None
        updated_at = None
        if data.get("createdAt"):
            created_at = datetime.fromisoformat(data["createdAt"].replace('Z', '+00:00'))
        if data.get("updatedAt"):
            updated_at = datetime.fromisoformat(data["updatedAt"].replace('Z', '+00:00'))
        
        return Contact(
            id=data.get("id"),
            first_name=data.get("firstName"),
            last_name=data.get("lastName"),
            email=data.get("email"),
            phone=data.get("phone"),
            address=data.get("address"),
            city=data.get("city"),
            state=data.get("state"),
            zip_code=data.get("postalCode"),
            country=data.get("country"),
            company=data.get("company"),
            website=data.get("website"),
            lead_source=data.get("source"),
            lead_score=lead_score,
            tags=data.get("tags", []),
            custom_fields=custom_fields,
            notes=data.get("notes"),
            status=ContactStatus(data.get("status", "new")) if data.get("status") else ContactStatus.NEW,
            pipeline_stage=PipelineStage(data.get("pipelineStage")) if data.get("pipelineStage") else None,
            assigned_user_id=data.get("assignedUser"),
            created_at=created_at,
            updated_at=updated_at
        )

    def _dict_to_pipeline(self, data: Dict[str, Any]) -> Pipeline:
        """Convert API response to Pipeline object."""
        created_at = None
        updated_at = None
        if data.get("createdAt"):
            created_at = datetime.fromisoformat(data["createdAt"].replace('Z', '+00:00'))
        if data.get("updatedAt"):
            updated_at = datetime.fromisoformat(data["updatedAt"].replace('Z', '+00:00'))
        
        return Pipeline(
            id=data["id"],
            name=data["name"],
            stages=data.get("stages", []),
            created_at=created_at,
            updated_at=updated_at
        )

    def _dict_to_tag(self, data: Dict[str, Any]) -> Tag:
        """Convert API response to Tag object."""
        created_at = None
        if data.get("createdAt"):
            created_at = datetime.fromisoformat(data["createdAt"].replace('Z', '+00:00'))
        
        return Tag(
            id=data["id"],
            name=data["name"],
            color=data.get("color"),
            created_at=created_at
        )

    # Async Methods for High Performance
    async def create_contact_async(self, contact: Contact) -> Contact:
        """Async version of create_contact."""
        if not self._async_session:
            raise GoHighLevelError("Async session not initialized. Use async context manager.")
        
        url = f"{self.base_url}/contacts/"
        data = self._contact_to_dict(contact)
        
        try:
            async with self._async_session.post(url, json=data) as response:
                if response.status == 401:
                    raise AuthenticationError("Invalid API key or authentication failed")
                elif response.status == 429:
                    raise RateLimitError("Rate limit exceeded")
                
                result = await response.json()
                
                if result.get("contact"):
                    return self._dict_to_contact(result["contact"])
                else:
                    raise GoHighLevelError("Contact creation failed")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Error creating contact async: {e}")
            raise GoHighLevelError(f"Failed to create contact async: {e}")

    async def batch_create_contacts_async(self, contacts: List[Contact], batch_size: int = 50) -> List[Contact]:
        """Batch create contacts asynchronously.
        
        Args:
            contacts: List of contacts to create
            batch_size: Number of contacts to process in parallel
            
        Returns:
            List of created contacts
        """
        if not self._async_session:
            raise GoHighLevelError("Async session not initialized. Use async context manager.")
        
        created_contacts = []
        
        for i in range(0, len(contacts), batch_size):
            batch = contacts[i:i + batch_size]
            tasks = [self.create_contact_async(contact) for contact in batch]
            
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in batch contact creation: {result}")
                    else:
                        created_contacts.append(result)
                        
            except Exception as e:
                logger.error(f"Error in batch contact creation: {e}")
        
        return created_contacts

    # Utility Methods
    def test_connection(self) -> bool:
        """Test API connection.
        
        Returns:
            True if connection is successful
        """
        try:
            self.get_tags()
            logger.info("GoHighLevel API connection test successful")
            return True
        except Exception as e:
            logger.error(f"GoHighLevel API connection test failed: {e}")
            return False

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information from response headers.
        
        Returns:
            Rate limit information
        """
        # This would typically be extracted from response headers
        # For now, return the configured rate limit
        return {
            "requests_per_second": 1.0 / self.rate_limit_delay,
            "delay_between_requests": self.rate_limit_delay,
            "max_retries": self.config.max_retries
        }