"""
Twilio Integration Package

This package provides comprehensive Twilio integration for Hail Hero, including:
- SMS/MMS messaging with consent management
- Voice calling capabilities
- Webhook handling for inbound communications
- Compliance and consent tracking
- Message templates and automation
"""

from .client import (
    TwilioClient,
    MessageType,
    MessageStatus,
    CallStatus,
    ConsentStatus,
    Message,
    Call,
    WebhookEvent,
    ConsentRecord,
    TwilioError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    ConsentError
)

__all__ = [
    'TwilioClient',
    'MessageType',
    'MessageStatus',
    'CallStatus',
    'ConsentStatus',
    'Message',
    'Call',
    'WebhookEvent',
    'ConsentRecord',
    'TwilioError',
    'AuthenticationError',
    'RateLimitError',
    'ValidationError',
    'ConsentError'
]