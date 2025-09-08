"""
Pytest configuration and fixtures for Hail Hero testing suite.
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_config, get_noaa_config, get_database_config
from src.noaa_api_client import NOAAAPIClient, APIResponse, APIError
from src.integrations.gohighlevel.client import GoHighLevelClient, Contact
from src.mvp.app import app as flask_app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        yield temp_path


@pytest.fixture
def test_config(temp_data_dir):
    """Create test configuration."""
    config = get_config()
    config.noaa.data_directory = str(temp_data_dir / "data")
    config.noaa.cache_directory = str(temp_data_dir / "cache")
    config.noaa.enable_caching = True
    config.noaa.cache_ttl_hours = 1
    config.debug = True
    config.environment = "testing"
    
    # Create directories
    (temp_data_dir / "data").mkdir(exist_ok=True)
    (temp_data_dir / "cache").mkdir(exist_ok=True)
    
    return config


@pytest.fixture
def mock_noaa_client():
    """Create a mock NOAA API client for testing."""
    client = Mock(spec=NOAAAPIClient)
    
    # Mock successful response
    mock_response = APIResponse(
        success=True,
        data=[
            {
                "EVENT_ID": "12345",
                "EVENT_TYPE": "Hail",
                "MAGNITUDE": 2.5,
                "BEGIN_DATE_TIME": "2023-06-15T14:30:00",
                "END_DATE_TIME": "2023-06-15T14:45:00",
                "BEGIN_LAT": 43.0642,
                "BEGIN_LON": -89.4005,
                "STATE": "WISCONSIN",
                "CZ_TYPE": "C",
                "CZ_NAME": "DANE"
            }
        ],
        metadata={"count": 1, "total": 1}
    )
    
    client.fetch_events = AsyncMock(return_value=mock_response)
    client.validate_token = AsyncMock(return_value=True)
    client.health_check = AsyncMock(return_value={
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "token_valid": True,
        "session_active": True
    })
    
    return client


@pytest.fixture
def mock_gohighlevel_client():
    """Create a mock GoHighLevel client for testing."""
    client = Mock(spec=GoHighLevelClient)
    
    # Mock contact methods
    client.create_contact = Mock(return_value=Contact(
        id="test_contact_123",
        first_name="John",
        last_name="Doe",
        email="john.doe@example.com",
        phone="+1234567890",
        address="123 Test St",
        city="Madison",
        state="WI",
        zip_code="53703"
    ))
    
    client.get_contact = Mock(return_value=Contact(
        id="test_contact_123",
        first_name="John",
        last_name="Doe",
        email="john.doe@example.com"
    ))
    
    client.test_connection = Mock(return_value=True)
    
    return client


@pytest.fixture
def sample_noaa_data():
    """Sample NOAA storm event data for testing."""
    return [
        {
            "EVENT_ID": "12345",
            "EVENT_TYPE": "Hail",
            "MAGNITUDE": 2.5,
            "BEGIN_DATE_TIME": "2023-06-15T14:30:00",
            "END_DATE_TIME": "2023-06-15T14:45:00",
            "BEGIN_LAT": 43.0642,
            "BEGIN_LON": -89.4005,
            "END_LAT": 43.0742,
            "END_LON": -89.4105,
            "STATE": "WISCONSIN",
            "CZ_TYPE": "C",
            "CZ_NAME": "DANE",
            "SOURCE": "Official",
            "EPISODE_ID": "67890"
        },
        {
            "EVENT_ID": "67890",
            "EVENT_TYPE": "Thunderstorm Wind",
            "MAGNITUDE": 75.0,
            "BEGIN_DATE_TIME": "2023-06-16T16:00:00",
            "END_DATE_TIME": "2023-06-16T16:15:00",
            "BEGIN_LAT": 42.7604,
            "BEGIN_LON": -89.2522,
            "STATE": "WISCONSIN",
            "CZ_TYPE": "C",
            "CZ_NAME": "ROCK",
            "SOURCE": "Official",
            "EPISODE_ID": "67891"
        }
    ]


@pytest.fixture
def sample_lead_data():
    """Sample lead data for testing."""
    return [
        {
            "lead_id": "LEAD_001",
            "event": {
                "EVENT_ID": "12345",
                "EVENT_TYPE": "Hail",
                "MAGNITUDE": 2.5,
                "BEGIN_DATE_TIME": "2023-06-15T14:30:00",
                "BEGIN_LAT": 43.0642,
                "BEGIN_LON": -89.4005,
                "STATE": "WISCONSIN"
            },
            "property": {
                "lat": 43.0642,
                "lon": -89.4005,
                "address": "123 Main St",
                "city": "Madison",
                "state": "WI",
                "zip_code": "53703"
            },
            "score": 25,
            "status": "new",
            "created_ts": "2023-06-15T15:00:00",
            "scoring_details": {
                "magnitude_score": 10,
                "proximity_score": 8,
                "recency_score": 7
            }
        },
        {
            "lead_id": "LEAD_002",
            "event": {
                "EVENT_ID": "67890",
                "EVENT_TYPE": "Thunderstorm Wind",
                "MAGNITUDE": 75.0,
                "BEGIN_DATE_TIME": "2023-06-16T16:00:00",
                "BEGIN_LAT": 42.7604,
                "BEGIN_LON": -89.2522,
                "STATE": "WISCONSIN"
            },
            "property": {
                "lat": 42.7604,
                "lon": -89.2522,
                "address": "456 Oak St",
                "city": "Janesville",
                "state": "WI",
                "zip_code": "53545"
            },
            "score": 18,
            "status": "inspected",
            "created_ts": "2023-06-16T17:00:00",
            "inspection": {
                "timestamp": "2023-06-16T18:00:00",
                "notes": "Property inspected, minor damage found",
                "photos": ["photo1.jpg", "photo2.jpg"]
            },
            "scoring_details": {
                "magnitude_score": 8,
                "proximity_score": 6,
                "recency_score": 4
            }
        }
    ]


@pytest.fixture
def sample_contact_data():
    """Sample contact data for testing."""
    return [
        {
            "id": "contact_123",
            "firstName": "John",
            "lastName": "Doe",
            "email": "john.doe@example.com",
            "phone": "+1234567890",
            "address": "123 Main St",
            "city": "Madison",
            "state": "WI",
            "postalCode": "53703",
            "source": "Hail Hero",
            "tags": ["hail_damage", "high_priority"],
            "customFields": {
                "lead_score": 25,
                "event_type": "Hail",
                "event_date": "2023-06-15"
            }
        },
        {
            "id": "contact_456",
            "firstName": "Jane",
            "lastName": "Smith",
            "email": "jane.smith@example.com",
            "phone": "+1234567891",
            "address": "456 Oak St",
            "city": "Janesville",
            "state": "WI",
            "postalCode": "53545",
            "source": "Hail Hero",
            "tags": ["wind_damage", "medium_priority"],
            "customFields": {
                "lead_score": 18,
                "event_type": "Thunderstorm Wind",
                "event_date": "2023-06-16"
            }
        }
    ]


@pytest.fixture
def flask_test_client():
    """Create a test client for the Flask app."""
    flask_app.config['TESTING'] = True
    flask_app.config['WTF_CSRF_ENABLED'] = False
    
    with flask_app.test_client() as client:
        with flask_app.app_context():
            yield client


@pytest.fixture
def auth_headers():
    """Mock authentication headers for testing."""
    return {
        "Authorization": "Bearer test_token",
        "Content-Type": "application/json"
    }


@pytest.fixture
def sample_inspection_data():
    """Sample inspection data for testing."""
    return {
        "lead_id": "LEAD_001",
        "notes": "Property inspected, found hail damage on roof and siding",
        "photos": [
            "roof_damage.jpg",
            "siding_damage.jpg",
            "closeup_hail.jpg"
        ],
        "damage_assessment": {
            "roof_damage": "moderate",
            "siding_damage": "minor",
            "window_damage": "none",
            "estimated_repair_cost": 8500
        },
        "recommendations": [
            "Roof replacement recommended",
            "Siding repair needed",
            "Gutter inspection required"
        ]
    }


@pytest.fixture
def api_error_response():
    """Sample API error response for testing."""
    return APIError(
        error_type="rate_limit_error",
        message="Rate limit exceeded",
        status_code=429,
        retry_after=60,
        details={"limit": 100, "window": "1h"}
    )


@pytest.fixture
def performance_metrics():
    """Sample performance metrics for testing."""
    return {
        "response_time": 0.234,
        "memory_usage": 45.2,
        "cpu_usage": 12.5,
        "request_count": 100,
        "error_rate": 0.02,
        "throughput": 150.5
    }


@pytest.fixture
def health_check_data():
    """Sample health check data for testing."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "uptime": 86400,
        "components": {
            "database": {"status": "healthy", "response_time": 0.012},
            "noaa_api": {"status": "healthy", "response_time": 0.234},
            "gohighlevel_api": {"status": "healthy", "response_time": 0.156},
            "cache": {"status": "healthy", "hit_rate": 0.85}
        },
        "metrics": {
            "total_requests": 1250,
            "error_rate": 0.015,
            "avg_response_time": 0.145
        }
    }


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slower, external deps)"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests (HTTP endpoints)"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (load, stress)"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests (full workflow)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (skip in CI)"
    )


# Test utilities
def create_test_lead_file(temp_data_dir, lead_data):
    """Create a test leads.jsonl file."""
    leads_file = temp_data_dir / "leads.jsonl"
    with open(leads_file, 'w') as f:
        for lead in lead_data:
            f.write(json.dumps(lead) + '\n')
    return leads_file


def create_test_event_file(temp_data_dir, event_data):
    """Create a test events.json file."""
    events_file = temp_data_dir / "events.json"
    with open(events_file, 'w') as f:
        json.dump(event_data, f, indent=2)
    return events_file


def assert_valid_lead(lead):
    """Assert that a lead object has valid structure."""
    assert "lead_id" in lead
    assert "event" in lead
    assert "property" in lead
    assert "score" in lead
    assert "status" in lead
    assert "created_ts" in lead
    
    # Validate event structure
    event = lead["event"]
    assert "EVENT_ID" in event
    assert "EVENT_TYPE" in event
    assert "MAGNITUDE" in event
    assert "BEGIN_DATE_TIME" in event
    assert "BEGIN_LAT" in event
    assert "BEGIN_LON" in event
    
    # Validate property structure
    prop = lead["property"]
    assert "lat" in prop
    assert "lon" in prop
    assert isinstance(prop["lat"], (int, float))
    assert isinstance(prop["lon"], (int, float))
    
    # Validate score
    assert isinstance(lead["score"], (int, float))
    assert lead["score"] >= 0
    
    # Validate status
    assert lead["status"] in ["new", "inspected", "qualified"]


def assert_valid_contact(contact):
    """Assert that a contact object has valid structure."""
    assert "id" in contact
    assert "firstName" in contact or "first_name" in contact
    assert "email" in contact
    assert isinstance(contact["id"], str)
    assert contact["id"]
    
    if "email" in contact:
        assert "@" in contact["email"]


def assert_api_response(response, expected_status=200):
    """Assert that an API response has the expected status and structure."""
    assert response.status_code == expected_status
    
    if response.headers.get('Content-Type', '').startswith('application/json'):
        data = response.get_json()
        assert isinstance(data, dict)
        return data
    
    return response.get_data(as_text=True)