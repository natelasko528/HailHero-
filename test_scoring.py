#!/usr/bin/env python3
"""Test script to demonstrate enhanced lead scoring algorithm."""

import sys
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mvp.runner import calculate_enhanced_lead_score, make_lead_from_event

def test_scoring_examples():
    """Test the enhanced scoring algorithm with various scenarios."""
    
    # Test cases representing different scenarios
    test_cases = [
        {
            'name': 'High-Value Chicago Area Hail (Peak Season)',
            'event': {
                'EVENT_ID': 'test-1',
                'EVENT_TYPE': 'Hail',
                'MAGNITUDE': 2.5,  # Large hail
                'BEGIN_LAT': 41.9,
                'BEGIN_LON': -87.7,  # Chicago area
                'STATE': 'IL',
                'BEGIN_DATE_TIME': '2025-06-15T14:00:00Z'  # Peak season
            }
        },
        {
            'name': 'Milwaukee Area Severe Hail',
            'event': {
                'EVENT_ID': 'test-2',
                'EVENT_TYPE': 'Hail',
                'MAGNITUDE': 3.0,  # Very large hail
                'BEGIN_LAT': 43.0,
                'BEGIN_LON': -88.0,  # Milwaukee area
                'STATE': 'WI',
                'BEGIN_DATE_TIME': '2025-07-20T16:30:00Z'  # Peak season
            }
        },
        {
            'name': 'Madison Area Moderate Hail',
            'event': {
                'EVENT_ID': 'test-3',
                'EVENT_TYPE': 'Hail',
                'MAGNITUDE': 1.8,  # Moderate hail
                'BEGIN_LAT': 43.1,
                'BEGIN_LON': -89.4,  # Madison area
                'STATE': 'WI',
                'BEGIN_DATE_TIME': '2025-05-10T15:00:00Z'  # Spring season
            }
        },
        {
            'name': 'Rural WI Small Hail',
            'event': {
                'EVENT_ID': 'test-4',
                'EVENT_TYPE': 'Hail',
                'MAGNITUDE': 0.8,  # Small hail
                'BEGIN_LAT': 44.5,
                'BEGIN_LON': -92.0,  # Rural area
                'STATE': 'WI',
                'BEGIN_DATE_TIME': '2025-09-15T12:00:00Z'  # Fall season
            }
        },
        {
            'name': 'High Wind Event',
            'event': {
                'EVENT_ID': 'test-5',
                'EVENT_TYPE': 'Thunderstorm Wind',
                'MAGNITUDE': 80,  # High wind
                'BEGIN_LAT': 42.0,
                'BEGIN_LON': -88.5,  # Northern IL
                'STATE': 'IL',
                'BEGIN_DATE_TIME': '2025-07-01T18:00:00Z'  # Peak season
            }
        },
        {
            'name': 'Tornado Event',
            'event': {
                'EVENT_ID': 'test-6',
                'EVENT_TYPE': 'Tornado',
                'MAGNITUDE': 2.0,  # EF-2 tornado
                'BEGIN_LAT': 42.8,
                'BEGIN_LON': -90.5,  # Central WI
                'STATE': 'WI',
                'BEGIN_DATE_TIME': '2025-06-01T20:00:00Z'  # Peak season
            }
        }
    ]
    
    print("Enhanced Lead Scoring Algorithm Test Results")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 50)
        
        # Calculate scores
        final_score, component_scores, confidence, tier = calculate_enhanced_lead_score(test_case['event'])
        
        # Generate full lead
        lead = make_lead_from_event(test_case['event'])
        
        print(f"Final Score: {final_score:.2f}")
        print(f"Tier: {tier.value.upper()}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Magnitude: {lead['event']['MAGNITUDE']}")
        print(f"Location: {lead['property']['lat']:.3f}, {lead['property']['lon']:.3f}")
        
        print("\nComponent Scores:")
        for component, score in component_scores.items():
            print(f"  {component.replace('_', ' ').title()}: {score:.2f}")
        
        print(f"\nLead Record:")
        print(f"  Lead ID: {lead['lead_id']}")
        print(f"  Event ID: {lead['event_id']}")
        print(f"  Status: {lead['status']}")
        
        # Print scoring details as JSON for better readability
        print(f"  Scoring Details: {json.dumps(lead['scoring_details'], indent=2)}")
    
    print("\n" + "=" * 60)
    print("Scoring Algorithm Summary:")
    print("- Event Severity (35%): Exponential scaling for hail, logarithmic for wind")
    print("- Geographic Location (25%): Proximity to high-value metropolitan areas")
    print("- Property Value (20%): Urban density and location-based estimation")
    print("- Seasonal Factors (10%): Peak hail season (May-August) gets higher scores")
    print("- Historical Patterns (10%): Areas with higher historical storm frequency")
    print("- Confidence Score: Based on data quality and consistency")
    print("- Tiers: Hot (80+), Warm (60-79), Cool (40-59), Cold (<40)")

if __name__ == '__main__':
    test_scoring_examples()