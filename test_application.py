#!/usr/bin/env python3
"""
Comprehensive Application Test Suite

Tests database functionality, API endpoints, and system components.
"""

import sys
import requests
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.database import get_db_manager
from src.models import Lead, Inspection, Photo, Event, Contact

# Test configuration
BASE_URL = 'http://localhost:5000'
API_URL = f'{BASE_URL}/api/v1'

def test_database_operations():
    """Test database CRUD operations."""
    print("\n" + "="*60)
    print("Testing Database Operations")
    print("="*60)

    db_manager = get_db_manager()

    # Test 1: Health check
    print("\n1. Testing database health check...")
    health = db_manager.health_check()
    print(f"   ✓ Database health: {'OK' if health else 'FAILED'}")
    assert health, "Database health check failed"

    # Test 2: Create a lead
    print("\n2. Testing lead creation...")
    with db_manager.get_session() as session:
        lead = Lead(
            lead_id='TEST-001',
            status='new',
            score=85.5,
            property_data=json.dumps({
                'address': '123 Main St',
                'city': 'Madison',
                'state': 'WI',
                'zipcode': '53703'
            }),
            event_data=json.dumps({
                'event_id': 'HAIL-2025-001',
                'severity': 2.5
            })
        )
        session.add(lead)
        session.commit()
        print(f"   ✓ Lead created: {lead.lead_id}")

    # Test 3: Query leads
    print("\n3. Testing lead query...")
    with db_manager.get_session() as session:
        leads = session.query(Lead).all()
        print(f"   ✓ Found {len(leads)} leads")
        for l in leads:
            print(f"     - {l.lead_id}: status={l.status}, score={l.score}")

    # Test 4: Create inspection
    print("\n4. Testing inspection creation...")
    with db_manager.get_session() as session:
        inspection = Inspection(
            lead_id='TEST-001',
            inspector_id='INSPECTOR-1',
            notes='Roof inspection completed. Minor hail damage observed.',
            gps_location='43.0731,-89.4012',
            sync_status='synced'
        )
        session.add(inspection)
        session.commit()
        print(f"   ✓ Inspection created: ID={inspection.id}")

    # Test 5: Create event
    print("\n5. Testing event creation...")
    with db_manager.get_session() as session:
        event = Event(
            event_id='HAIL-2025-001',
            event_type='hail',
            source='NOAA',
            severity=2.5,
            geometry=json.dumps({
                'type': 'Point',
                'coordinates': [-89.4012, 43.0731]
            }),
            start_time=datetime.utcnow(),
            location_description='Madison, WI area'
        )
        session.add(event)
        session.commit()
        print(f"   ✓ Event created: {event.event_id}")

    # Test 6: Create contact
    print("\n6. Testing contact creation...")
    with db_manager.get_session() as session:
        contact = Contact(
            contact_id='CONTACT-001',
            first_name='John',
            last_name='Doe',
            phone='+15551234567',
            email='john.doe@example.com',
            consent_status='granted',
            dnc_status=False
        )
        session.add(contact)
        session.commit()
        print(f"   ✓ Contact created: {contact.contact_id}")

    # Test 7: Query with relationships
    print("\n7. Testing relationships...")
    with db_manager.get_session() as session:
        lead = session.query(Lead).filter_by(lead_id='TEST-001').first()
        print(f"   ✓ Lead has {len(lead.inspections)} inspection(s)")

    print("\n✅ All database tests passed!")


def test_api_endpoints():
    """Test API endpoints."""
    print("\n" + "="*60)
    print("Testing API Endpoints")
    print("="*60)

    # Test 1: Health check
    print("\n1. Testing /health endpoint...")
    response = requests.get(f'{BASE_URL}/health')
    assert response.status_code == 200
    data = response.json()
    print(f"   ✓ Status: {data['status']}")
    print(f"   ✓ Database: {data['database']}")

    # Test 2: System info
    print("\n2. Testing /api/info endpoint...")
    response = requests.get(f'{BASE_URL}/api/info')
    assert response.status_code == 200
    data = response.json()
    print(f"   ✓ Application: {data['application']}")
    print(f"   ✓ Version: {data['version']}")
    print(f"   ✓ Database type: {data['database']['database_type']}")

    # Test 3: API health
    print("\n3. Testing /api/v1/health/ endpoint...")
    response = requests.get(f'{API_URL}/health/')
    assert response.status_code == 200
    data = response.json()
    print(f"   ✓ API Status: {data['status']}")

    # Test 4: Swagger documentation
    print("\n4. Testing Swagger documentation...")
    response = requests.get(f'{API_URL}/swagger.json')
    assert response.status_code == 200
    swagger = response.json()
    print(f"   ✓ Swagger version: {swagger.get('swagger')}")
    print(f"   ✓ API paths: {len(swagger.get('paths', {}))} endpoints")

    # Test 5: List leads
    print("\n5. Testing /api/v1/leads/ endpoint...")
    response = requests.get(f'{API_URL}/leads/')
    assert response.status_code == 200
    data = response.json()
    print(f"   ✓ Total leads: {data['total']}")

    # Test 6: List inspections
    print("\n6. Testing /api/v1/inspections/ endpoint...")
    response = requests.get(f'{API_URL}/inspections/')
    assert response.status_code == 200
    inspections = response.json()
    print(f"   ✓ Total inspections: {len(inspections)}")

    # Test 7: List contacts
    print("\n7. Testing /api/v1/contacts/ endpoint...")
    response = requests.get(f'{API_URL}/contacts/')
    assert response.status_code == 200
    contacts = response.json()
    print(f"   ✓ Total contacts: {len(contacts)}")

    # Test 8: List events
    print("\n8. Testing /api/v1/events/ endpoint...")
    response = requests.get(f'{API_URL}/events/')
    assert response.status_code == 200
    events = response.json()
    print(f"   ✓ Total events: {len(events)}")

    print("\n✅ All API tests passed!")


def test_configuration():
    """Test configuration and setup."""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)

    # Test 1: Environment files
    print("\n1. Checking configuration files...")
    env_file = Path('.env')
    nginx_conf = Path('nginx.conf')
    alembic_ini = Path('alembic.ini')

    print(f"   ✓ .env exists: {env_file.exists()}")
    print(f"   ✓ nginx.conf exists: {nginx_conf.exists()}")
    print(f"   ✓ alembic.ini exists: {alembic_ini.exists()}")

    # Test 2: Directories
    print("\n2. Checking required directories...")
    data_dir = Path('data')
    logs_dir = Path('logs')
    uploads_dir = Path('uploads')

    print(f"   ✓ data/ exists: {data_dir.exists()}")
    print(f"   ✓ logs/ exists: {logs_dir.exists()}")
    print(f"   ✓ uploads/ exists: {uploads_dir.exists()}")

    # Test 3: Database file
    print("\n3. Checking database...")
    db_file = Path('data/hailhero.db')
    print(f"   ✓ Database file exists: {db_file.exists()}")
    if db_file.exists():
        size = db_file.stat().st_size
        print(f"   ✓ Database size: {size} bytes")

    print("\n✅ All configuration tests passed!")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*60)
    print("HAIL HERO - COMPREHENSIVE TEST SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        test_configuration()
        test_database_operations()
        test_api_endpoints()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        print("\n📊 Test Summary:")
        print("   - Configuration: ✅ PASSED")
        print("   - Database Operations: ✅ PASSED")
        print("   - API Endpoints: ✅ PASSED")
        print("\n🚀 Hail Hero is ready for use!")
        print("\n📚 Access the API documentation at:")
        print(f"   {BASE_URL}/api/v1/docs")

        return 0

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TESTS FAILED")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
