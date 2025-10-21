#!/usr/bin/env python3
"""
Enhanced Hail Hero Application Runner

Starts the Flask application with:
- API documentation (Swagger/OpenAPI)
- Database abstraction layer
- Health checks
- CORS support
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

# Import modules
from src.database import get_db_manager, init_db
from src.mvp.api_docs import init_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure logging
if not app.debug:
    handler = RotatingFileHandler('logs/hailhero.log', maxBytes=10000000, backupCount=3)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

# Initialize database
logger.info("Initializing database...")
db_manager = get_db_manager()
logger.info(f"Database connection: {db_manager.get_connection_info()}")

# Register API documentation
logger.info("Registering API documentation...")
init_api(app)

# Root endpoint
@app.route('/')
def index():
    """Redirect to API documentation."""
    return redirect('/api/v1/docs')

# Health check endpoint
@app.route('/health')
def health():
    """System health check."""
    db_health = db_manager.health_check()
    return jsonify({
        'status': 'healthy' if db_health else 'unhealthy',
        'database': 'connected' if db_health else 'disconnected',
        'version': '1.0.0'
    }), 200 if db_health else 503

# Database info endpoint
@app.route('/api/info')
def info():
    """Get system information."""
    return jsonify({
        'application': 'Hail Hero',
        'version': '1.0.0',
        'database': db_manager.get_connection_info(),
        'endpoints': {
            'api_docs': '/api/v1/docs',
            'health': '/health',
            'swagger': '/api/v1/swagger.json'
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': str(error)}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Internal error: {error}')
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

if __name__ == '__main__':
    # Get configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting Hail Hero application on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"API Documentation available at: http://{host}:{port}/api/v1/docs")

    app.run(host=host, port=port, debug=debug)
