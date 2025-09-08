"""
Twilio Integration Module for Hail Hero

This module provides comprehensive Twilio integration for SMS/MMS messaging,
voice calls, and webhook handling with proper error handling and consent management.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import phonenumbers
from phonenumbers.phonenumberutil import NumberParseException

from ..integrations.config import get_twilio_config, TwilioConfig
from ..compliance.models import ConsentManager, DataProvenance

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types supported by Twilio integration."""
    SMS = "sms"
    MMS = "mms"
    VOICE = "voice"
    WHATSAPP = "whatsapp"


class MessageStatus(Enum):
    """Message status values."""
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    UNDELIVERED = "undelivered"
    FAILED = "failed"
    RECEIVED = "received"


class CallStatus(Enum):
    """Call status values."""
    QUEUED = "queued"
    RINGING = "ringing"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BUSY = "busy"
    NO_ANSWER = "no-answer"


class ConsentStatus(Enum):
    """Consent status for messaging."""
    OPTED_IN = "opted_in"
    OPTED_OUT = "opted_out"
    PENDING = "pending"
    EXPIRED = "expired"


@dataclass
class Message:
    """Twilio message data model."""
    sid: Optional[str] = None
    from_phone: Optional[str] = None
    to_phone: Optional[str] = None
    body: Optional[str] = None
    status: MessageStatus = MessageStatus.QUEUED
    message_type: MessageType = MessageType.SMS
    media_url: Optional[str] = None
    direction: str = "outbound"  # outbound, inbound
    date_created: Optional[datetime] = None
    date_sent: Optional[datetime] = None
    date_updated: Optional[datetime] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    price: Optional[str] = None
    num_segments: Optional[int] = None
    num_media: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Call:
    """Twilio call data model."""
    sid: Optional[str] = None
    from_phone: Optional[str] = None
    to_phone: Optional[str] = None
    status: CallStatus = CallStatus.QUEUED
    direction: str = "outbound"  # outbound, inbound
    duration: Optional[int] = None  # in seconds
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    price: Optional[str] = None
    recording_url: Optional[str] = None
    transcription: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WebhookEvent:
    """Twilio webhook event data model."""
    event_type: str
    message_sid: Optional[str] = None
    call_sid: Optional[str] = None
    from_phone: Optional[str] = None
    to_phone: Optional[str] = None
    body: Optional[str] = None
    status: Optional[str] = None
    timestamp: Optional[datetime] = None
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass
class ConsentRecord:
    """Consent record for messaging compliance."""
    phone_number: str
    status: ConsentStatus
    consent_method: str  # sms, voice, webform, etc.
    consent_timestamp: datetime
    expiration_timestamp: Optional[datetime] = None
    campaign_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TwilioError(Exception):
    """Base exception for Twilio integration errors."""
    pass


class AuthenticationError(TwilioError):
    """Authentication related errors."""
    pass


class RateLimitError(TwilioError):
    """Rate limit exceeded errors."""
    pass


class ValidationError(TwilioError):
    """Data validation errors."""
    pass


class ConsentError(TwilioError):
    """Consent related errors."""
    pass


class TwilioClient:
    """Twilio API client with comprehensive functionality."""

    def __init__(self, account_sid: Optional[str] = None, auth_token: Optional[str] = None):
        """Initialize Twilio client.
        
        Args:
            account_sid: Twilio account SID
            auth_token: Twilio auth token
        """
        self.config = get_twilio_config()
        
        self.account_sid = account_sid or self.config.account_sid
        self.auth_token = auth_token or self.config.auth_token
        self.phone_number = self.config.phone_number
        
        if not self.account_sid:
            raise AuthenticationError("Account SID is required")
        
        if not self.auth_token:
            raise AuthenticationError("Auth token is required")
        
        # API endpoints
        self.base_url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}"
        self.messages_url = f"{self.base_url}/Messages.json"
        self.calls_url = f"{self.base_url}/Calls.json"
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1.0,
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
        
        # Consent management
        self.consent_manager = ConsentManager()
        self._consent_cache: Dict[str, ConsentRecord] = {}
        
        # Message templates
        self.message_templates = {
            'hail_alert': "🚨 Hail Alert: Severe hail detected in your area! {hail_size}\" hail reported. Property inspection recommended. Reply STOP to unsubscribe.",
            'lead_followup': "Hi {name}, following up on the recent hail activity in your area. Would you like a free roof inspection? Reply YES or STOP to unsubscribe.",
            'appointment_reminder': "Reminder: Your roof inspection is scheduled for {date} at {time}. Reply CANCEL to reschedule.",
            'thank_you': "Thank you for choosing Hail Hero! Your inspection report is ready. View it here: {link}"
        }
        
        logger.info(f"Twilio client initialized for account: {self.account_sid}")

    async def __aenter__(self):
        """Async context manager entry."""
        self._async_session = aiohttp.ClientSession(
            auth=aiohttp.BasicAuth(self.account_sid, self.auth_token),
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._async_session:
            await self._async_session.close()

    def _handle_rate_limit(self):
        """Handle rate limiting with delay."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _validate_phone_number(self, phone_number: str) -> str:
        """Validate and format phone number."""
        try:
            parsed = phonenumbers.parse(phone_number, self.config.default_country_code)
            
            if not phonenumbers.is_valid_number(parsed):
                raise ValidationError(f"Invalid phone number: {phone_number}")
            
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
            
        except NumberParseException as e:
            raise ValidationError(f"Invalid phone number format: {phone_number} - {e}")

    def _check_consent(self, phone_number: str) -> ConsentStatus:
        """Check consent status for a phone number."""
        # Check cache first
        if phone_number in self._consent_cache:
            record = self._consent_cache[phone_number]
            
            # Check if consent has expired
            if (record.expiration_timestamp and 
                datetime.utcnow() > record.expiration_timestamp):
                record.status = ConsentStatus.EXPIRED
                self._consent_cache[phone_number] = record
            
            return record.status
        
        # Check database (this would be implemented with your ORM)
        # For now, default to pending
        return ConsentStatus.PENDING

    def _add_consent_record(self, phone_number: str, status: ConsentStatus, 
                           consent_method: str, campaign_id: Optional[str] = None):
        """Add or update consent record."""
        record = ConsentRecord(
            phone_number=phone_number,
            status=status,
            consent_method=consent_method,
            consent_timestamp=datetime.utcnow(),
            expiration_timestamp=datetime.utcnow() + timedelta(days=365),  # 1 year
            campaign_id=campaign_id,
            metadata={}
        )
        
        self._consent_cache[phone_number] = record
        
        # Here you would also save to database
        logger.info(f"Consent record updated for {phone_number}: {status.value}")

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response with error handling."""
        if response.status_code == 401:
            raise AuthenticationError("Invalid account SID or auth token")
        elif response.status_code == 403:
            raise AuthenticationError("Access forbidden - check permissions")
        elif response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        elif response.status_code == 400:
            raise ValidationError(f"Bad request: {response.text}")
        elif response.status_code >= 500:
            raise TwilioError(f"Server error: {response.status_code}")
        
        try:
            return response.json()
        except json.JSONDecodeError:
            if response.status_code >= 400:
                raise TwilioError(f"HTTP {response.status_code}: {response.text}")
            return {"status": "success"}

    async def send_message(self, to_phone: str, message: str, 
                          message_type: MessageType = MessageType.SMS,
                          media_url: Optional[str] = None,
                          campaign_id: Optional[str] = None,
                          require_consent: bool = True) -> Dict[str, Any]:
        """Send SMS/MMS message.
        
        Args:
            to_phone: Recipient phone number
            message: Message content
            message_type: Type of message
            media_url: URL for MMS media
            campaign_id: Campaign ID for tracking
            require_consent: Whether to require consent
            
        Returns:
            Message result data
        """
        try:
            # Validate phone number
            to_phone = self._validate_phone_number(to_phone)
            
            # Check consent if required
            if require_consent and self.config.require_consent:
                consent_status = self._check_consent(to_phone)
                if consent_status == ConsentStatus.OPTED_OUT:
                    raise ConsentError(f"Recipient {to_phone} has opted out")
                elif consent_status == ConsentStatus.PENDING:
                    logger.warning(f"Sending message to {to_phone} with pending consent")
            
            # Add consent text if required
            if self.config.consent_text and not message.endswith(self.config.consent_text):
                message = f"{message}\n\n{self.config.consent_text}"
            
            # Truncate message if too long
            if len(message) > self.config.max_message_length:
                message = message[:self.config.max_message_length-3] + "..."
            
            # Prepare message data
            message_data = {
                "From": self.phone_number,
                "To": to_phone,
                "Body": message
            }
            
            if media_url and message_type == MessageType.MMS:
                if not self.config.enable_mms:
                    raise ValidationError("MMS is not enabled")
                message_data["MediaUrl"] = media_url
            
            # Send message
            self._handle_rate_limit()
            
            response = self.session.post(
                self.messages_url,
                data=message_data,
                auth=(self.account_sid, self.auth_token),
                timeout=self.config.timeout
            )
            
            result = self._handle_response(response)
            
            # Add consent record if this was the first message
            if require_consent and self._check_consent(to_phone) == ConsentStatus.PENDING:
                self._add_consent_record(to_phone, ConsentStatus.OPTED_IN, 'sms', campaign_id)
            
            # Log for compliance
            DataProvenance.log_message(
                from_phone=self.phone_number,
                to_phone=to_phone,
                message=message,
                message_type=message_type.value,
                campaign_id=campaign_id
            )
            
            logger.info(f"Message sent to {to_phone}: {result.get('sid')}")
            
            return {
                "sid": result.get("sid"),
                "status": result.get("status"),
                "to_phone": to_phone,
                "message_type": message_type.value,
                "sent_at": datetime.utcnow().isoformat(),
                "campaign_id": campaign_id
            }
            
        except Exception as e:
            logger.error(f"Error sending message to {to_phone}: {e}")
            raise

    async def send_template_message(self, to_phone: str, template_name: str, 
                                   template_data: Dict[str, Any],
                                   message_type: MessageType = MessageType.SMS,
                                   campaign_id: Optional[str] = None) -> Dict[str, Any]:
        """Send message using a template.
        
        Args:
            to_phone: Recipient phone number
            template_name: Name of template to use
            template_data: Data for template substitution
            message_type: Type of message
            campaign_id: Campaign ID for tracking
            
        Returns:
            Message result data
        """
        if template_name not in self.message_templates:
            raise ValidationError(f"Unknown template: {template_name}")
        
        # Format template with data
        message = self.message_templates[template_name].format(**template_data)
        
        return await self.send_message(
            to_phone=to_phone,
            message=message,
            message_type=message_type,
            campaign_id=campaign_id
        )

    async def make_call(self, to_phone: str, twiml_url: str, 
                       campaign_id: Optional[str] = None) -> Dict[str, Any]:
        """Make outbound call.
        
        Args:
            to_phone: Recipient phone number
            twiml_url: URL for TwiML instructions
            campaign_id: Campaign ID for tracking
            
        Returns:
            Call result data
        """
        try:
            # Validate phone number
            to_phone = self._validate_phone_number(to_phone)
            
            # Prepare call data
            call_data = {
                "From": self.phone_number,
                "To": to_phone,
                "Url": twiml_url,
                "Method": "POST"
            }
            
            # Make call
            self._handle_rate_limit()
            
            response = self.session.post(
                self.calls_url,
                data=call_data,
                auth=(self.account_sid, self.auth_token),
                timeout=self.config.timeout
            )
            
            result = self._handle_response(response)
            
            logger.info(f"Call initiated to {to_phone}: {result.get('sid')}")
            
            return {
                "sid": result.get("sid"),
                "status": result.get("status"),
                "to_phone": to_phone,
                "call_started_at": datetime.utcnow().isoformat(),
                "campaign_id": campaign_id
            }
            
        except Exception as e:
            logger.error(f"Error making call to {to_phone}: {e}")
            raise

    async def get_message(self, message_sid: str) -> Message:
        """Get message details by SID.
        
        Args:
            message_sid: Message SID
            
        Returns:
            Message object
        """
        try:
            url = f"{self.messages_url}/{message_sid}.json"
            
            response = self.session.get(
                url,
                auth=(self.account_sid, self.auth_token),
                timeout=self.config.timeout
            )
            
            result = self._handle_response(response)
            
            return self._dict_to_message(result)
            
        except Exception as e:
            logger.error(f"Error getting message {message_sid}: {e}")
            raise

    async def get_call(self, call_sid: str) -> Call:
        """Get call details by SID.
        
        Args:
            call_sid: Call SID
            
        Returns:
            Call object
        """
        try:
            url = f"{self.calls_url}/{call_sid}.json"
            
            response = self.session.get(
                url,
                auth=(self.account_sid, self.auth_token),
                timeout=self.config.timeout
            )
            
            result = self._handle_response(response)
            
            return self._dict_to_call(result)
            
        except Exception as e:
            logger.error(f"Error getting call {call_sid}: {e}")
            raise

    async def get_message_history(self, to_phone: Optional[str] = None, 
                                 limit: int = 50) -> List[Message]:
        """Get message history.
        
        Args:
            to_phone: Filter by recipient phone number
            limit: Maximum number of results
            
        Returns:
            List of Message objects
        """
        try:
            params = {"Limit": limit}
            
            if to_phone:
                params["To"] = self._validate_phone_number(to_phone)
            
            response = self.session.get(
                self.messages_url,
                params=params,
                auth=(self.account_sid, self.auth_token),
                timeout=self.config.timeout
            )
            
            result = self._handle_response(response)
            
            messages = []
            for message_data in result.get("messages", []):
                messages.append(self._dict_to_message(message_data))
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting message history: {e}")
            raise

    def parse_webhook_event(self, payload: Dict[str, Any]) -> WebhookEvent:
        """Parse Twilio webhook payload.
        
        Args:
            payload: Raw webhook payload
            
        Returns:
            Parsed WebhookEvent object
        """
        try:
            # Determine event type
            event_type = "message_received"
            if payload.get("CallSid"):
                event_type = "call_event"
            elif payload.get("MessageStatus"):
                event_type = "message_status"
            
            # Parse timestamp
            timestamp = None
            if payload.get("Timestamp"):
                timestamp = datetime.fromisoformat(payload["Timestamp"].replace('Z', '+00:00'))
            
            return WebhookEvent(
                event_type=event_type,
                message_sid=payload.get("MessageSid"),
                call_sid=payload.get("CallSid"),
                from_phone=payload.get("From"),
                to_phone=payload.get("To"),
                body=payload.get("Body"),
                status=payload.get("MessageStatus") or payload.get("CallStatus"),
                timestamp=timestamp,
                raw_payload=payload
            )
            
        except Exception as e:
            logger.error(f"Error parsing webhook event: {e}")
            raise ValidationError(f"Failed to parse webhook event: {e}")

    async def handle_webhook_event(self, event: WebhookEvent) -> bool:
        """Handle incoming webhook event.
        
        Args:
            event: Parsed webhook event
            
        Returns:
            True if processed successfully
        """
        try:
            logger.info(f"Processing webhook event: {event.event_type}")
            
            if event.event_type == "message_received":
                return await self._handle_inbound_message(event)
            elif event.event_type == "message_status":
                return await self._handle_message_status(event)
            elif event.event_type == "call_event":
                return await self._handle_call_event(event)
            else:
                logger.warning(f"Unhandled webhook event type: {event.event_type}")
                return True
                
        except Exception as e:
            logger.error(f"Error handling webhook event: {e}")
            return False

    async def _handle_inbound_message(self, event: WebhookEvent) -> bool:
        """Handle inbound message."""
        if not event.body:
            return False
        
        # Check for opt-out/opt-in keywords
        body_lower = event.body.lower().strip()
        
        if body_lower in ['stop', 'unsubscribe', 'cancel']:
            # Opt-out
            self._add_consent_record(event.from_phone, ConsentStatus.OPTED_OUT, 'sms')
            
            # Send confirmation
            await self.send_message(
                to_phone=event.from_phone,
                message="You have been unsubscribed. Reply START to resubscribe.",
                require_consent=False
            )
            
            logger.info(f"Opt-out received from {event.from_phone}")
            
        elif body_lower in ['start', 'subscribe', 'yes']:
            # Opt-in
            self._add_consent_record(event.from_phone, ConsentStatus.OPTED_IN, 'sms')
            
            # Send confirmation
            await self.send_message(
                to_phone=event.from_phone,
                message="You have been subscribed. Reply STOP to unsubscribe.",
                require_consent=False
            )
            
            logger.info(f"Opt-in received from {event.from_phone}")
        
        # Log for compliance
        DataProvenance.log_message(
            from_phone=event.from_phone,
            to_phone=event.to_phone,
            message=event.body,
            message_type="inbound_sms",
            campaign_id=None
        )
        
        return True

    async def _handle_message_status(self, event: WebhookEvent) -> bool:
        """Handle message status update."""
        logger.info(f"Message status update: {event.message_sid} - {event.status}")
        
        # Here you would update your database with the new status
        # This is useful for tracking message delivery
        
        return True

    async def _handle_call_event(self, event: WebhookEvent) -> bool:
        """Handle call event."""
        logger.info(f"Call event: {event.call_sid} - {event.status}")
        
        # Here you would handle call events (ringing, answered, etc.)
        
        return True

    def _dict_to_message(self, data: Dict[str, Any]) -> Message:
        """Convert API response to Message object."""
        # Parse dates
        date_created = None
        date_sent = None
        date_updated = None
        
        if data.get("date_created"):
            date_created = datetime.fromisoformat(data["date_created"].replace('Z', '+00:00'))
        if data.get("date_sent"):
            date_sent = datetime.fromisoformat(data["date_sent"].replace('Z', '+00:00'))
        if data.get("date_updated"):
            date_updated = datetime.fromisoformat(data["date_updated"].replace('Z', '+00:00'))
        
        # Determine message type
        message_type = MessageType.SMS
        if data.get("num_media", 0) > 0:
            message_type = MessageType.MMS
        
        # Determine status
        try:
            status = MessageStatus(data.get("status", "queued"))
        except ValueError:
            status = MessageStatus.QUEUED
        
        return Message(
            sid=data.get("sid"),
            from_phone=data.get("from"),
            to_phone=data.get("to"),
            body=data.get("body"),
            status=status,
            message_type=message_type,
            media_url=data.get("media_url"),
            direction=data.get("direction", "outbound"),
            date_created=date_created,
            date_sent=date_sent,
            date_updated=date_updated,
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            price=data.get("price"),
            num_segments=data.get("num_segments"),
            num_media=data.get("num_media")
        )

    def _dict_to_call(self, data: Dict[str, Any]) -> Call:
        """Convert API response to Call object."""
        # Parse dates
        start_time = None
        end_time = None
        
        if data.get("start_time"):
            start_time = datetime.fromisoformat(data["start_time"].replace('Z', '+00:00'))
        if data.get("end_time"):
            end_time = datetime.fromisoformat(data["end_time"].replace('Z', '+00:00'))
        
        # Determine status
        try:
            status = CallStatus(data.get("status", "queued"))
        except ValueError:
            status = CallStatus.QUEUED
        
        return Call(
            sid=data.get("sid"),
            from_phone=data.get("from"),
            to_phone=data.get("to"),
            status=status,
            direction=data.get("direction", "outbound"),
            duration=data.get("duration"),
            start_time=start_time,
            end_time=end_time,
            price=data.get("price"),
            recording_url=data.get("recording_url"),
            transcription=data.get("transcription")
        )

    def get_consent_status(self, phone_number: str) -> ConsentStatus:
        """Get consent status for a phone number.
        
        Args:
            phone_number: Phone number to check
            
        Returns:
            Consent status
        """
        return self._check_consent(phone_number)

    def get_consent_records(self, limit: int = 100) -> List[ConsentRecord]:
        """Get consent records.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            List of consent records
        """
        # Return cached records (in production, this would query the database)
        return list(self._consent_cache.values())[:limit]

    def add_message_template(self, name: str, template: str):
        """Add a custom message template.
        
        Args:
            name: Template name
            template: Template string with placeholders
        """
        self.message_templates[name] = template
        logger.info(f"Added message template: {name}")

    def get_message_templates(self) -> Dict[str, str]:
        """Get all message templates.
        
        Returns:
            Dictionary of template names and content
        """
        return self.message_templates.copy()

    def test_connection(self) -> bool:
        """Test Twilio API connection.
        
        Returns:
            True if connection is successful
        """
        try:
            # Try to get account info
            response = self.session.get(
                self.base_url + ".json",
                auth=(self.account_sid, self.auth_token),
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                logger.info("Twilio API connection test successful")
                return True
            else:
                logger.error(f"Twilio API connection test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Twilio API connection test error: {e}")
            return False

    def get_usage_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics.
        
        Args:
            days: Number of days to include in stats
            
        Returns:
            Usage statistics
        """
        # This would typically use Twilio's usage API
        # For now, return placeholder data
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            },
            "messages_sent": 0,
            "messages_received": 0,
            "calls_made": 0,
            "total_cost": 0.0,
            "consent_rate": 0.0
        }