#!/usr/bin/env python3
"""
Create sample data for testing and demonstration
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.database import get_db_manager
from src.models import Lead, Inspection, Photo, Event, Contact

def create_sample_data():
    """Create realistic sample data for demonstration."""
    print("\n🎨 Creating sample data for Hail Hero...\n")

    db_manager = get_db_manager()

    with db_manager.get_session() as session:
        # Create sample hail events
        events = [
            Event(
                event_id='NOAA-WI-2025-001',
                event_type='hail',
                source='NOAA',
                severity=2.5,
                geometry=json.dumps({
                    'type': 'Point',
                    'coordinates': [-89.4012, 43.0731]
                }),
                start_time=datetime.utcnow() - timedelta(days=5),
                location_description='Madison, WI - Dane County',
                event_metadata=json.dumps({
                    'hail_size': '2.5 inches',
                    'wind_speed': '65 mph',
                    'damage_estimate': 'moderate'
                })
            ),
            Event(
                event_id='NOAA-IL-2025-002',
                event_type='hail',
                source='NOAA',
                severity=3.0,
                geometry=json.dumps({
                    'type': 'Point',
                    'coordinates': [-87.6298, 41.8781]
                }),
                start_time=datetime.utcnow() - timedelta(days=3),
                location_description='Chicago, IL - Cook County',
                event_metadata=json.dumps({
                    'hail_size': '3.0 inches',
                    'wind_speed': '75 mph',
                    'damage_estimate': 'severe'
                })
            )
        ]

        for event in events:
            session.add(event)

        print(f"✓ Created {len(events)} hail events")

        # Create sample contacts
        contacts = [
            Contact(
                contact_id='CONTACT-WI-001',
                first_name='Sarah',
                last_name='Johnson',
                phone='+16085551234',
                email='sarah.johnson@example.com',
                address='123 Oak Street, Madison, WI 53703',
                consent_status='granted',
                consent_timestamp=datetime.utcnow() - timedelta(days=2),
                dnc_status=False,
                source_provenance=json.dumps({
                    'source': 'OpenAddresses',
                    'confidence': 0.95,
                    'retrieved_at': datetime.utcnow().isoformat()
                })
            ),
            Contact(
                contact_id='CONTACT-IL-001',
                first_name='Michael',
                last_name='Chen',
                phone='+13125555678',
                email='michael.chen@example.com',
                address='456 Maple Avenue, Chicago, IL 60614',
                consent_status='granted',
                consent_timestamp=datetime.utcnow() - timedelta(days=1),
                dnc_status=False,
                source_provenance=json.dumps({
                    'source': 'OpenAddresses',
                    'confidence': 0.92,
                    'retrieved_at': datetime.utcnow().isoformat()
                })
            ),
            Contact(
                contact_id='CONTACT-WI-002',
                first_name='Emily',
                last_name='Rodriguez',
                phone='+16085559012',
                email='emily.rodriguez@example.com',
                address='789 Pine Road, Madison, WI 53704',
                consent_status='pending',
                dnc_status=False,
                source_provenance=json.dumps({
                    'source': 'OpenStreetMap',
                    'confidence': 0.88,
                    'retrieved_at': datetime.utcnow().isoformat()
                })
            )
        ]

        for contact in contacts:
            session.add(contact)

        print(f"✓ Created {len(contacts)} contacts")

        # Create sample leads
        leads = [
            Lead(
                lead_id='LEAD-2025-001',
                status='new',
                score=92.5,
                property_data=json.dumps({
                    'address': '123 Oak Street',
                    'city': 'Madison',
                    'state': 'WI',
                    'zipcode': '53703',
                    'latitude': 43.0731,
                    'longitude': -89.4012,
                    'roof_type': 'Asphalt Shingle',
                    'roof_age': 8,
                    'square_footage': 2400
                }),
                event_data=json.dumps({
                    'event_id': 'NOAA-WI-2025-001',
                    'severity': 2.5,
                    'distance_from_center': 0.5
                }),
                scoring_details=json.dumps({
                    'event_severity_score': 85,
                    'property_age_score': 90,
                    'location_score': 100,
                    'overall_score': 92.5,
                    'factors': ['Recent construction', 'Direct hit', 'Visible damage likely']
                })
            ),
            Lead(
                lead_id='LEAD-2025-002',
                status='contacted',
                score=88.0,
                property_data=json.dumps({
                    'address': '456 Maple Avenue',
                    'city': 'Chicago',
                    'state': 'IL',
                    'zipcode': '60614',
                    'latitude': 41.8781,
                    'longitude': -87.6298,
                    'roof_type': 'Metal',
                    'roof_age': 5,
                    'square_footage': 3200
                }),
                event_data=json.dumps({
                    'event_id': 'NOAA-IL-2025-002',
                    'severity': 3.0,
                    'distance_from_center': 1.2
                }),
                scoring_details=json.dumps({
                    'event_severity_score': 95,
                    'property_age_score': 85,
                    'location_score': 80,
                    'overall_score': 88.0,
                    'factors': ['New roof', 'Severe event', 'High value property']
                })
            ),
            Lead(
                lead_id='LEAD-2025-003',
                status='inspected',
                score=95.0,
                property_data=json.dumps({
                    'address': '789 Pine Road',
                    'city': 'Madison',
                    'state': 'WI',
                    'zipcode': '53704',
                    'latitude': 43.0950,
                    'longitude': -89.3850,
                    'roof_type': 'Composite',
                    'roof_age': 12,
                    'square_footage': 2800
                }),
                event_data=json.dumps({
                    'event_id': 'NOAA-WI-2025-001',
                    'severity': 2.5,
                    'distance_from_center': 0.3
                }),
                scoring_details=json.dumps({
                    'event_severity_score': 85,
                    'property_age_score': 95,
                    'location_score': 100,
                    'overall_score': 95.0,
                    'factors': ['Older roof', 'Direct impact', 'Confirmed damage']
                })
            )
        ]

        for lead in leads:
            session.add(lead)

        print(f"✓ Created {len(leads)} leads")

        # Commit to get IDs
        session.commit()

        # Create sample inspections
        inspections = [
            Inspection(
                lead_id='LEAD-2025-003',
                inspector_id='INSPECTOR-JD-001',
                notes="""Inspection completed on 10/18/2025.

Findings:
- Multiple impact points on north-facing roof section
- Approximately 15-20 dents ranging from 1-2 inches
- Some granule loss visible
- Fascia damage on northwest corner
- Recommend full roof replacement

Photos: 4 overview shots, 8 close-ups of damage
Estimated claim value: $12,000-$15,000""",
                photos=json.dumps(['roof_overview_1.jpg', 'damage_closeup_1.jpg', 'damage_closeup_2.jpg']),
                gps_location='43.0950,-89.3850',
                timestamp=datetime.utcnow() - timedelta(hours=2),
                sync_status='synced'
            ),
            Inspection(
                lead_id='LEAD-2025-002',
                inspector_id='INSPECTOR-SM-002',
                notes="""Initial inspection scheduled for 10/22/2025.

Pre-inspection notes:
- Homeowner reports hearing loud impact sounds during storm
- Visible debris in gutters
- Metal roof may show denting
- Schedule 2-hour inspection window

Status: Scheduled""",
                gps_location='41.8781,-87.6298',
                timestamp=datetime.utcnow() - timedelta(hours=24),
                sync_status='pending'
            )
        ]

        for inspection in inspections:
            session.add(inspection)

        print(f"✓ Created {len(inspections)} inspections")

        session.commit()

    print("\n✅ Sample data created successfully!")
    print("\n📊 Summary:")
    print(f"   • {len(events)} Hail Events")
    print(f"   • {len(contacts)} Contacts")
    print(f"   • {len(leads)} Leads")
    print(f"   • {len(inspections)} Inspections")
    print("\n🔗 View data at: http://localhost:5000/api/v1/docs\n")


if __name__ == '__main__':
    create_sample_data()
