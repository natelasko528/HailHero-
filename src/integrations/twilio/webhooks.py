"""
Webhook handlers for Twilio integration.

This module provides Flask webhook endpoints for handling:
- Incoming SMS/MMS messages
- Incoming voice calls
- Call status updates
- Message status updates
- Consent management webhooks
"""

import json
import logging
import hashlib
import hmac
from datetime import datetime
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify, Response, abort
from functools import wraps

from .client import TwilioClient, MessageStatus, CallStatus, ConsentStatus
from .config import get_twilio_config

logger = logging.getLogger(__name__)


class WebhookSecurityError(Exception):
    """Security error for webhook validation."""
    pass


class TwilioWebhookHandler:
    """Handler for Twilio webhooks."""
    
    def __init__(self, app: Flask, twilio_client: TwilioClient):
        """Initialize webhook handler."""
        self.app = app
        self.twilio_client = twilio_client
        self.config = get_twilio_config()
        
        # Register webhook routes
        self._register_routes()
        
        logger.info("Twilio webhook handler initialized")
    
    def _register_routes(self):
        """Register Flask routes for webhooks."""
        # SMS webhook
        self.app.route('/api/twilio/sms', methods=['POST'])(self.handle_sms_webhook)
        
        # Voice webhook
        self.app.route('/api/twilio/voice', methods=['POST'])(self.handle_voice_webhook)
        
        # Call status webhook
        self.app.route('/api/twilio/call-status', methods=['POST'])(self.handle_call_status_webhook)
        
        # Message status webhook
        self.app.route('/api/twilio/message-status', methods=['POST'])(self.handle_message_status_webhook)
        
        # Health check
        self.app.route('/api/twilio/health', methods=['GET'])(self.health_check)
    
    def validate_twilio_request(self, f):
        """Decorator to validate Twilio webhook requests."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                self._validate_request_signature(request)
                return f(*args, **kwargs)
            except WebhookSecurityError as e:
                logger.error(f"Webhook validation failed: {e}")
                abort(403)
            except Exception as e:
                logger.error(f"Webhook validation error: {e}")
                abort(500)
        return decorated_function
    
    def _validate_request_signature(self, request):
        """Validate Twilio request signature."""
        if not self.config.webhook_secret:
            # Skip validation if no secret is configured (development mode)
            logger.warning("No webhook secret configured - skipping validation")
            return
        
        # Get the Twilio signature
        signature = request.headers.get('X-Twilio-Signature', '')
        if not signature:
            raise WebhookSecurityError("Missing X-Twilio-Signature header")
        
        # Get the URL (including query parameters)
        url = request.url
        
        # Get the POST parameters
        params = request.form.to_dict()
        
        # Validate the signature
        expected_signature = self._calculate_signature(url, params)
        
        if not hmac.compare_digest(signature, expected_signature):
            raise WebhookSecurityError("Invalid signature")
    
    def _calculate_signature(self, url: str, params: Dict[str, str]) -> str:
        """Calculate expected signature for validation."""
        # Sort parameters
        sorted_params = sorted(params.items())
        
        # Create parameter string
        param_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
        
        # Create full string to sign
        full_string = f"{url}{param_string}"
        
        # Calculate signature
        signature = hmac.new(
            self.config.webhook_secret.encode('utf-8'),
            full_string.encode('utf-8'),
            hashlib.sha1
        ).hexdigest()
        
        return signature
    
    @validate_twilio_request
    def handle_sms_webhook(self):
        """Handle incoming SMS webhook."""
        try:
            # Extract webhook data
            from_number = request.form.get('From', '')
            to_number = request.form.get('To', '')
            body = request.form.get('Body', '')
            message_sid = request.form.get('MessageSid', '')
            num_media = int(request.form.get('NumMedia', 0))
            
            # Extract media URLs if present
            media_urls = []
            for i in range(num_media):
                media_url = request.form.get(f'MediaUrl{i}', '')
                if media_url:
                    media_urls.append(media_url)
            
            logger.info(f"Processing incoming SMS from {from_number}: {body[:100]}...")
            
            # Handle the incoming message
            message = self.twilio_client.handle_incoming_sms(
                from_number=from_number,
                to_number=to_number,
                content=body,
                media_urls=media_urls if media_urls else None
            )
            
            # Log the webhook processing
            self._log_webhook_event(
                event_type='sms_received',
                from_number=from_number,
                to_number=to_number,
                message_sid=message_sid,
                body=body,
                media_count=len(media_urls)
            )
            
            # Generate response if needed (auto-reply)
            response = self._generate_sms_response(message)
            
            return Response(response, mimetype='text/xml')
            
        except Exception as e:
            logger.error(f"Error handling SMS webhook: {e}")
            # Return empty response to avoid retry
            return Response('', mimetype='text/xml')
    
    def _generate_sms_response(self, message) -> str:
        """Generate automated SMS response."""
        content = message.content.upper().strip()
        
        # Handle common commands
        if content in ["STOP", "STOPALL", "UNSUBSCRIBE"]:
            return self.twilio_client.generate_twiml_sms_response(
                "You have been unsubscribed. Reply START to resubscribe."
            )
        elif content in ["START", "YES", "SUBSCRIBE"]:
            return self.twilio_client.generate_twiml_sms_response(
                "You have been subscribed. Reply STOP to unsubscribe."
            )
        elif content == "HELP":
            return self.twilio_client.generate_twiml_sms_response(
                "Reply STOP to unsubscribe, START to subscribe, or HELP for this message."
            )
        
        # Default response for other messages
        return self.twilio_client.generate_twiml_sms_response(
            "Thank you for your message. Our team will get back to you soon."
        )
    
    @validate_twilio_request
    def handle_voice_webhook(self):
        """Handle incoming voice call webhook."""
        try:
            # Extract webhook data
            from_number = request.form.get('From', '')
            to_number = request.form.get('To', '')
            call_sid = request.form.get('CallSid', '')
            call_status = request.form.get('CallStatus', '')
            direction = request.form.get('Direction', '')
            
            logger.info(f"Processing incoming voice call from {from_number}: {call_sid}")
            
            # Handle the incoming call
            call = self._handle_incoming_call(
                from_number=from_number,
                to_number=to_number,
                call_sid=call_sid,
                call_status=call_status,
                direction=direction
            )
            
            # Log the webhook processing
            self._log_webhook_event(
                event_type='call_received',
                from_number=from_number,
                to_number=to_number,
                call_sid=call_sid,
                call_status=call_status,
                direction=direction
            )
            
            # Generate TwiML response
            response = self._generate_voice_response(call)
            
            return Response(response, mimetype='text/xml')
            
        except Exception as e:
            logger.error(f"Error handling voice webhook: {e}")
            # Return empty response to avoid retry
            return Response('', mimetype='text/xml')
    
    def _handle_incoming_call(self, from_number: str, to_number: str, call_sid: str,
                            call_status: str, direction: str):
        """Handle incoming voice call."""
        try:
            # Format phone numbers
            from_formatted = self.twilio_client.format_phone_number(from_number)
            to_formatted = self.twilio_client.format_phone_number(to_number)
            
            # Check consent
            consent_status = self.twilio_client.check_consent(from_formatted)
            if consent_status == ConsentStatus.REVOKED:
                logger.warning(f"Rejected call from DNC number: {from_formatted}")
                # Return TwiML to reject the call
                return self.twilio_client.generate_twiml_voice_response(
                    "This number is not accepting calls at this time."
                )
            
            # Create call record
            call = self.twilio_client.Call(
                id=f"incoming_call_{int(datetime.now().timestamp())}",
                from_number=from_formatted,
                to_number=to_formatted,
                status=CallStatus.RINGING,
                twilio_sid=call_sid,
                direction=direction
            )
            
            # Add to call history
            self.twilio_client.call_history.append(call)
            
            logger.info(f"Incoming call processed: {call.id} from {from_formatted}")
            
            return call
            
        except Exception as e:
            logger.error(f"Error handling incoming call: {e}")
            raise
    
    def _generate_voice_response(self, call) -> str:
        """Generate TwiML response for voice call."""
        # Generate greeting message
        greeting = "Thank you for calling Hail Hero. Please hold while we connect you to a representative."
        
        # In a real implementation, you would:
        # 1. Look up the caller in your database
        # 2. Check if they have any open leads
        # 3. Route to appropriate agent or IVR
        # 4. Record the call for quality assurance
        
        return self.twilio_client.generate_twiml_voice_response(greeting)
    
    @validate_twilio_request
    def handle_call_status_webhook(self):
        """Handle call status update webhook."""
        try:
            # Extract webhook data
            call_sid = request.form.get('CallSid', '')
            call_status = request.form.get('CallStatus', '')
            call_duration = request.form.get('CallDuration', '')
            recording_url = request.form.get('RecordingUrl', '')
            
            logger.info(f"Processing call status update: {call_sid} -> {call_status}")
            
            # Update call status
            self._update_call_status(
                call_sid=call_sid,
                call_status=call_status,
                call_duration=int(call_duration) if call_duration else None,
                recording_url=recording_url
            )
            
            # Log the webhook processing
            self._log_webhook_event(
                event_type='call_status_update',
                call_sid=call_sid,
                call_status=call_status,
                call_duration=call_duration,
                recording_url=recording_url
            )
            
            return '', 204
            
        except Exception as e:
            logger.error(f"Error handling call status webhook: {e}")
            return '', 500
    
    def _update_call_status(self, call_sid: str, call_status: str, 
                           call_duration: Optional[int] = None,
                           recording_url: Optional[str] = None):
        """Update call status in history."""
        try:
            # Find the call in history
            call = None
            for c in self.twilio_client.call_history:
                if c.twilio_sid == call_sid:
                    call = c
                    break
            
            if not call:
                logger.warning(f"Call not found for status update: {call_sid}")
                return
            
            # Map Twilio status to our enum
            status_mapping = {
                'queued': CallStatus.QUEUED,
                'ringing': CallStatus.RINGING,
                'in-progress': CallStatus.IN_PROGRESS,
                'completed': CallStatus.COMPLETED,
                'failed': CallStatus.FAILED,
                'busy': CallStatus.BUSY,
                'no-answer': CallStatus.NO_ANSWER
            }
            
            # Update call status
            call.status = status_mapping.get(call_status, CallStatus.FAILED)
            
            # Update additional fields
            if call_duration:
                call.duration = call_duration
            
            if recording_url:
                call.recording_url = recording_url
            
            # Update timestamps
            if call.status == CallStatus.IN_PROGRESS:
                call.started_at = datetime.now()
            elif call.status in [CallStatus.COMPLETED, CallStatus.FAILED, CallStatus.BUSY, CallStatus.NO_ANSWER]:
                call.ended_at = datetime.now()
            
            logger.info(f"Call status updated: {call.id} -> {call.status.value}")
            
        except Exception as e:
            logger.error(f"Error updating call status: {e}")
    
    @validate_twilio_request
    def handle_message_status_webhook(self):
        """Handle message status update webhook."""
        try:
            # Extract webhook data
            message_sid = request.form.get('MessageSid', '')
            message_status = request.form.get('MessageStatus', '')
            error_code = request.form.get('ErrorCode', '')
            error_message = request.form.get('ErrorMessage', '')
            
            logger.info(f"Processing message status update: {message_sid} -> {message_status}")
            
            # Update message status
            self._update_message_status(
                message_sid=message_sid,
                message_status=message_status,
                error_code=error_code,
                error_message=error_message
            )
            
            # Log the webhook processing
            self._log_webhook_event(
                event_type='message_status_update',
                message_sid=message_sid,
                message_status=message_status,
                error_code=error_code,
                error_message=error_message
            )
            
            return '', 204
            
        except Exception as e:
            logger.error(f"Error handling message status webhook: {e}")
            return '', 500
    
    def _update_message_status(self, message_sid: str, message_status: str,
                             error_code: Optional[str] = None,
                             error_message: Optional[str] = None):
        """Update message status in history."""
        try:
            # Find the message in history
            message = None
            for m in self.twilio_client.message_history:
                if m.twilio_sid == message_sid:
                    message = m
                    break
            
            if not message:
                logger.warning(f"Message not found for status update: {message_sid}")
                return
            
            # Map Twilio status to our enum
            status_mapping = {
                'queued': MessageStatus.QUEUED,
                'sent': MessageStatus.SENT,
                'delivered': MessageStatus.DELIVERED,
                'undelivered': MessageStatus.UNDELIVERED,
                'failed': MessageStatus.FAILED
            }
            
            # Update message status
            message.status = status_mapping.get(message_status, MessageStatus.QUEUED)
            
            # Update timestamps
            if message.status == MessageStatus.SENT:
                message.sent_at = datetime.now()
            elif message.status == MessageStatus.DELIVERED:
                message.delivered_at = datetime.now()
            
            # Update error information
            if error_code or error_message:
                message.error_message = f"Error {error_code}: {error_message}"
            
            logger.info(f"Message status updated: {message.id} -> {message.status.value}")
            
        except Exception as e:
            logger.error(f"Error updating message status: {e}")
    
    def _log_webhook_event(self, event_type: str, **kwargs):
        """Log webhook event for auditing."""
        try:
            event_data = {
                'event_type': event_type,
                'timestamp': datetime.now().isoformat(),
                'ip_address': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', ''),
                **kwargs
            }
            
            logger.info(f"Webhook event: {json.dumps(event_data)}")
            
            # In production, you might want to store this in a database
            # for auditing and analytics purposes
            
        except Exception as e:
            logger.error(f"Error logging webhook event: {e}")
    
    def health_check(self):
        """Health check endpoint."""
        try:
            # Check if Twilio client is configured
            if not self.twilio_client.client:
                return jsonify({
                    'status': 'error',
                    'message': 'Twilio client not configured'
                }), 500
            
            # Get queue status
            queue_status = self.twilio_client.get_queue_status()
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'queue_status': queue_status
            })
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    def get_webhook_stats(self) -> Dict[str, Any]:
        """Get webhook statistics."""
        try:
            # Count webhook events by type
            sms_count = len([m for m in self.twilio_client.message_history if m.direction == 'inbound'])
            call_count = len([c for c in self.twilio_client.call_history if c.direction == 'inbound'])
            
            # Get recent webhook activity
            recent_messages = [m for m in self.twilio_client.message_history 
                             if m.timestamp > datetime.now() - timedelta(hours=24)]
            recent_calls = [c for c in self.twilio_client.call_history 
                          if c.timestamp > datetime.now() - timedelta(hours=24)]
            
            return {
                'total_sms_received': sms_count,
                'total_calls_received': call_count,
                'recent_sms_count': len(recent_messages),
                'recent_calls_count': len(recent_calls),
                'consent_records': len(self.twilio_client.consent_records),
                'active_consent': len([r for r in self.twilio_client.consent_records.values() 
                                      if r.consent_status == ConsentStatus.CONSENTED])
            }
            
        except Exception as e:
            logger.error(f"Error getting webhook stats: {e}")
            return {}


# Flask blueprint for Twilio webhooks
def create_twilio_webhook_blueprint(twilio_client: TwilioClient):
    """Create Flask blueprint for Twilio webhooks."""
    from flask import Blueprint
    
    blueprint = Blueprint('twilio_webhooks', __name__)
    handler = TwilioWebhookHandler(blueprint, twilio_client)
    
    return blueprint