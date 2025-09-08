"""Twilio webhook prototype for Hail Hero.

This lightweight Flask app receives inbound Twilio webhooks (SMS/MMS/call events)
and logs them; it's a placeholder for integrating with a lead store and creating
interaction events. Intended for local development and testing only.
"""

import logging
from typing import Any, Dict
from flask.wrappers import Response

from flask import Flask, request, jsonify


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


@app.route('/twilio/webhook', methods=['POST'])
def twilio_webhook() -> Response:
    """Receive Twilio webhook POST and return a debug JSON response.

    This function intentionally keeps logic minimal: it logs the inbound payload
    and returns a simple JSON object. Replace placeholder sections with real
    lead-store lookups and idempotency checks when wiring into the application.
    """
    data = request.form.to_dict()
    message_sid = data.get('MessageSid')
    from_number = data.get('From')
    to_number = data.get('To')
    body = data.get('Body')

    app.logger.info(
        'Twilio webhook received: sid=%s from=%s to=%s',
        message_sid,
        from_number,
        to_number,
    )

    # Idempotency: log / store MessageSid and ignore duplicates (placeholder)

    # Match phone to lead (placeholder logic)
    # TODO: implement lookup in lead store and idempotent processing

    # Create interaction event (placeholder)
    interaction = {
        'message_sid': message_sid,
        'from': from_number,
        'to': to_number,
        'body': body,
    }

    # For now, return 200 OK with JSON for debugging
    return jsonify({'status': 'ok', 'interaction': interaction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
