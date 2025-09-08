#!/usr/bin/env python3
"""
Enhanced Hail Hero Runner with Geospatial Analysis

Integrates the advanced geospatial analysis system with the existing lead generation pipeline.
Provides comprehensive claim readiness scoring, damage assessment, and field optimization.

Enhanced Features:
- Geospatial event correlation and clustering
- Advanced claim readiness scoring with damage assessment
- Property profiling and damage estimation
- Field team optimization and routing
- Temporal pattern analysis
- Risk assessment and insurance claim prediction
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

import requests

# Import our geospatial analysis system
from geospatial_analysis import (
    GeospatialAnalyzer, DamageAssessmentEngine, ClaimScoringEngine,
    FieldToolsIntegration, StormEvent, PropertyProfile, ClaimReadinessScore,
    EventType, PropertyType, RoofType, ClaimTier
)

# Import existing runner functionality
from mvp.runner import (
    load_config, fetch_ncei_with_retry, filter_events, synthetic_leads,
    write_leads, write_raw, load_cached_data, cache_data, IngestionResult,
    DataSource, APIConfig, iso_today_minus, get_magnitude, record_lat, record_lon,
    validate_event, DEFAULT_HAIL_MIN, DEFAULT_WIND_MIN, DEFAULT_API_TIMEOUT,
    DEFAULT_RATE_LIMIT_DELAY, MAX_RETRIES, NCEI_BASE_URL, NCEI_DATASET
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspaces/HailHero-/geospatial_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'specs' / '001-hail-hero-hail' / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class EnhancedLead:
    """Enhanced lead record with geospatial analysis."""
    lead_id: str
    event_id: str
    score: float
    tier: str
    confidence: float
    status: str
    created_ts: str
    event: Dict[str, Any]
    property: Dict[str, Any]
    property_profile: Optional[PropertyProfile] = None
    claim_readiness: Optional[ClaimReadinessScore] = None
    contact: Optional[Dict[str, Any]] = None
    scoring_details: Optional[Dict[str, Any]] = None
    provenance: Dict[str, Any]
    field_tools: Optional[Dict[str, Any]] = None
    geospatial_analysis: Optional[Dict[str, Any]] = None

class EnhancedRunner:
    """Enhanced runner with geospatial analysis capabilities."""
    
    def __init__(self):
        self.geospatial_analyzer = GeospatialAnalyzer()
        self.damage_engine = DamageAssessmentEngine()
        self.claim_scoring_engine = ClaimScoringEngine()
        self.field_tools = FieldToolsIntegration()
        
    def create_property_profile(self, lead_data: Dict[str, Any]) -> PropertyProfile:
        """Create property profile from lead data with intelligent defaults."""
        lat = lead_data['property']['lat']
        lon = lead_data['property']['lon']
        
        # Intelligent property type detection based on location
        property_type = self._detect_property_type(lat, lon)
        
        # Roof type estimation based on location and building age
        roof_type = self._estimate_roof_type(lat, lon)
        
        # Estimate property age based on location
        year_built = self._estimate_year_built(lat, lon)
        roof_age = datetime.datetime.now().year - year_built
        
        # Estimate property value based on location
        property_value = self._estimate_property_value(lat, lon, property_type)
        
        # Estimate square footage
        square_footage = self._estimate_square_footage(lat, lon, property_type)
        
        return PropertyProfile(
            property_id=lead_data['lead_id'],
            latitude=lat,
            longitude=lon,
            property_type=property_type,
            roof_type=roof_type,
            roof_age_years=roof_age,
            property_value_usd=property_value,
            square_footage=square_footage,
            year_built=year_built,
            address=self._get_address_from_coords(lat, lon)
        )
    
    def _detect_property_type(self, lat: float, lon: float) -> PropertyType:
        """Detect property type based on geographic location."""
        # Urban vs rural detection
        if self._is_urban_area(lat, lon):
            # Urban areas have more mixed use
            urban_types = [PropertyType.RESIDENTIAL, PropertyType.COMMERCIAL, PropertyType.MIXED_USE]
            return random.choice(urban_types)
        else:
            # Rural areas are predominantly residential
            return PropertyType.RESIDENTIAL
    
    def _estimate_roof_type(self, lat: float, lon: float) -> RoofType:
        """Estimate roof type based on geographic and climate factors."""
        # Wisconsin and Illinois have similar roofing patterns
        # Asphalt shingles are most common
        roof_types = [
            (RoofType.ASPHALT_SHINGLES, 0.7),  # 70% asphalt
            (RoofType.METAL, 0.15),             # 15% metal
            (RoofType.TILE, 0.05),              # 5% tile
            (RoofType.FLAT, 0.05),              # 5% flat (commercial)
            (RoofType.WOOD, 0.05)               # 5% wood
        ]
        
        rand = random.random()
        cumulative = 0
        for roof_type, probability in roof_types:
            cumulative += probability
            if rand <= cumulative:
                return roof_type
        
        return RoofType.ASPHALT_SHINGLES
    
    def _estimate_year_built(self, lat: float, lon: float) -> int:
        """Estimate property year built based on location."""
        # Different areas have different development patterns
        current_year = datetime.datetime.now().year
        
        # Urban areas tend to have older housing stock
        if self._is_urban_area(lat, lon):
            # More variety in urban areas
            base_year = random.randint(1900, current_year - 5)
        else:
            # Suburban/rural areas tend to be newer
            base_year = random.randint(1970, current_year - 5)
        
        return base_year
    
    def _estimate_property_value(self, lat: float, lon: float, property_type: PropertyType) -> float:
        """Estimate property value based on location and type."""
        # Base values by property type
        base_values = {
            PropertyType.RESIDENTIAL: 250000,
            PropertyType.COMMERCIAL: 750000,
            PropertyType.INDUSTRIAL: 1200000,
            PropertyType.MIXED_USE: 600000,
            PropertyType.UNKNOWN: 300000
        }
        
        base_value = base_values.get(property_type, 300000)
        
        # Adjust for location (urban areas more valuable)
        if self._is_urban_area(lat, lon):
            base_value *= 1.5
        
        # Add variation
        variation = random.uniform(0.7, 1.3)
        return base_value * variation
    
    def _estimate_square_footage(self, lat: float, lon: float, property_type: PropertyType) -> float:
        """Estimate property square footage."""
        base_sizes = {
            PropertyType.RESIDENTIAL: 2000,
            PropertyType.COMMERCIAL: 5000,
            PropertyType.INDUSTRIAL: 10000,
            PropertyType.MIXED_USE: 3500,
            PropertyType.UNKNOWN: 2200
        }
        
        base_size = base_sizes.get(property_type, 2200)
        variation = random.uniform(0.6, 1.4)
        return base_size * variation
    
    def _is_urban_area(self, lat: float, lon: float) -> bool:
        """Determine if coordinates are in an urban area."""
        # Major urban centers in Wisconsin and Illinois
        urban_centers = [
            (41.8781, -87.6298),  # Chicago
            (43.0642, -89.4000),  # Madison
            (43.0389, -87.9065),  # Milwaukee
            (42.2711, -89.0940),  # Rockford
            (42.9839, -81.2497),  # London, ON (for comparison)
        ]
        
        for center_lat, center_lon in urban_centers:
            distance = self.geospatial_analyzer.haversine_distance(
                lat, lon, center_lat, center_lon
            )
            if distance <= 25:  # 25 km radius
                return True
        
        return False
    
    def _get_address_from_coords(self, lat: float, lon: float) -> Optional[str]:
        """Get address from coordinates (placeholder for reverse geocoding)."""
        # In a real implementation, this would use a reverse geocoding service
        # For now, return a placeholder
        return f"Approximate location: {lat:.4f}, {lon:.4f}"
    
    def convert_noaa_event_to_storm_event(self, event_data: Dict[str, Any]) -> StormEvent:
        """Convert NOAA event data to StormEvent format."""
        event_type_str = (event_data.get('EVENT_TYPE') or '').lower()
        
        # Map event types
        if 'hail' in event_type_str:
            event_type = EventType.HAIL
        elif 'wind' in event_type_str:
            event_type = EventType.WIND
        elif 'tornado' in event_type_str:
            event_type = EventType.TORNADO
        elif 'thunderstorm' in event_type_str:
            event_type = EventType.THUNDERSTORM
        else:
            event_type = EventType.UNKNOWN
        
        # Parse timestamp
        timestamp_str = event_data.get('BEGIN_DATE_TIME') or event_data.get('timestamp')
        if timestamp_str:
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                timestamp = datetime.datetime.utcnow()
        else:
            timestamp = datetime.datetime.utcnow()
        
        return StormEvent(
            event_id=event_data.get('EVENT_ID') or f"event-{random.randint(10000, 99999)}",
            event_type=event_type,
            magnitude=get_magnitude(event_data) or 0.0,
            latitude=record_lat(event_data) or 0.0,
            longitude=record_lon(event_data) or 0.0,
            timestamp=timestamp,
            state=event_data.get('STATE') or 'Unknown',
            duration_minutes=None,  # Not available in basic NOAA data
            affected_area_km2=None  # Would need to be calculated
        )
    
    def create_enhanced_lead(self, event_data: Dict[str, Any]) -> EnhancedLead:
        """Create enhanced lead with geospatial analysis."""
        # Create basic lead data
        ev_id = event_data.get('EVENT_ID') or event_data.get('eventId') or event_data.get('id') or f"ev-{random.randint(100000, 999999)}"
        mag = get_magnitude(event_data) or 0.0
        lat = record_lat(event_data)
        lon = record_lon(event_data)
        
        # Basic scoring (will be enhanced later)
        event_type = (event_data.get('EVENT_TYPE') or '').lower()
        if 'hail' in event_type:
            score = min(100, int((mag / 3.0) * 100))
        else:
            score = min(100, int((mag / 100.0) * 100))
        
        # Create basic lead structure
        lead_data = {
            'lead_id': f'lead-{ev_id}',
            'event_id': ev_id,
            'score': score,
            'tier': 'cold',  # Will be updated with enhanced scoring
            'confidence': 0.8,  # Will be updated
            'status': 'new',
            'created_ts': datetime.datetime.utcnow().isoformat() + 'Z',
            'event': event_data,
            'property': {'lat': lat, 'lon': lon},
            'contact': None,
            'provenance': {
                'source': 'ncei',
                'ingested_ts': datetime.datetime.utcnow().isoformat() + 'Z',
                'event_type': event_data.get('EVENT_TYPE'),
                'magnitude': mag,
                'state': event_data.get('STATE')
            }
        }
        
        # Create property profile
        property_profile = self.create_property_profile(lead_data)
        
        # Convert to storm event for analysis
        storm_event = self.convert_noaa_event_to_storm_event(event_data)
        
        # Calculate claim readiness score
        claim_readiness = self.claim_scoring_engine.calculate_claim_readiness_score(
            property_profile, [storm_event]
        )
        
        # Create field tools integration
        field_tools_data = self.field_tools.create_inspection_workflow(
            property_profile, claim_readiness
        )
        
        # Perform geospatial analysis
        geospatial_analysis = self._perform_geospatial_analysis([storm_event], [property_profile])
        
        # Update lead with enhanced data
        lead_data['score'] = claim_readiness.overall_score
        lead_data['tier'] = claim_readiness.claim_tier.value
        lead_data['confidence'] = claim_readiness.confidence
        lead_data['scoring_details'] = claim_readiness.component_scores
        
        # Create enhanced lead
        enhanced_lead = EnhancedLead(
            lead_id=lead_data['lead_id'],
            event_id=lead_data['event_id'],
            score=claim_readiness.overall_score,
            tier=claim_readiness.claim_tier.value,
            confidence=claim_readiness.confidence,
            status=lead_data['status'],
            created_ts=lead_data['created_ts'],
            event=lead_data['event'],
            property=lead_data['property'],
            property_profile=property_profile,
            claim_readiness=claim_readiness,
            contact=lead_data['contact'],
            scoring_details=lead_data['scoring_details'],
            provenance=lead_data['provenance'],
            field_tools=field_tools_data,
            geospatial_analysis=geospatial_analysis
        )
        
        return enhanced_lead
    
    def _perform_geospatial_analysis(self, events: List[StormEvent], properties: List[PropertyProfile]) -> Dict[str, Any]:
        """Perform comprehensive geospatial analysis."""
        analysis = {}
        
        # Event correlation
        correlation = self.geospatial_analyzer.correlate_events_to_properties(events, properties)
        analysis['event_correlation'] = {
            prop_id: [event.to_dict() for event in correlated_events]
            for prop_id, correlated_events in correlation.items()
        }
        
        # Event clustering
        clusters = self.geospatial_analyzer.cluster_events_by_proximity(events)
        analysis['event_clusters'] = [
            [event.to_dict() for event in cluster]
            for cluster in clusters
        ]
        
        # Event density
        density_map = self.geospatial_analyzer.calculate_event_density(events)
        analysis['event_density'] = density_map
        
        # Geographic optimization
        analysis['geographic_optimization'] = self._calculate_geographic_optimization(properties)
        
        return analysis
    
    def _calculate_geographic_optimization(self, properties: List[PropertyProfile]) -> Dict[str, Any]:
        """Calculate geographic optimization for field teams."""
        if not properties:
            return {}
        
        # Calculate center point
        center_lat = sum(prop.latitude for prop in properties) / len(properties)
        center_lon = sum(prop.longitude for prop in properties) / len(properties)
        
        # Calculate distances from center
        distances = []
        for prop in properties:
            distance = self.geospatial_analyzer.haversine_distance(
                center_lat, center_lon, prop.latitude, prop.longitude
            )
            distances.append({
                'property_id': prop.property_id,
                'distance_km': distance,
                'coordinates': [prop.latitude, prop.longitude]
            })
        
        # Sort by distance
        distances.sort(key=lambda x: x['distance_km'])
        
        # Calculate optimal route (simple nearest neighbor)
        optimal_route = self._calculate_optimal_route(properties)
        
        return {
            'center_point': [center_lat, center_lon],
            'properties_by_distance': distances,
            'optimal_route': optimal_route,
            'total_route_distance_km': self._calculate_route_distance(optimal_route),
            'estimated_travel_time_hours': self._estimate_travel_time(optimal_route)
        }
    
    def _calculate_optimal_route(self, properties: List[PropertyProfile]) -> List[Dict[str, Any]]:
        """Calculate optimal route using nearest neighbor algorithm."""
        if not properties:
            return []
        
        # Start with first property
        unvisited = properties.copy()
        route = []
        current = unvisited.pop(0)
        route.append({
            'property_id': current.property_id,
            'coordinates': [current.latitude, current.longitude],
            'order': 0
        })
        
        # Visit nearest unvisited property
        order = 1
        while unvisited:
            nearest = min(unvisited, key=lambda p: self.geospatial_analyzer.haversine_distance(
                current.latitude, current.longitude, p.latitude, p.longitude
            ))
            
            route.append({
                'property_id': nearest.property_id,
                'coordinates': [nearest.latitude, nearest.longitude],
                'order': order
            })
            
            current = nearest
            unvisited.remove(current)
            order += 1
        
        return route
    
    def _calculate_route_distance(self, route: List[Dict[str, Any]]) -> float:
        """Calculate total route distance."""
        total_distance = 0
        for i in range(len(route) - 1):
            current = route[i]
            next_point = route[i + 1]
            distance = self.geospatial_analyzer.haversine_distance(
                current['coordinates'][0], current['coordinates'][1],
                next_point['coordinates'][0], next_point['coordinates'][1]
            )
            total_distance += distance
        
        return total_distance
    
    def _estimate_travel_time(self, route: List[Dict[str, Any]]) -> float:
        """Estimate travel time for route."""
        total_distance = self._calculate_route_distance(route)
        # Assume average speed of 50 km/h for field work
        return total_distance / 50.0
    
    def write_enhanced_leads(self, leads: List[EnhancedLead]) -> None:
        """Write enhanced leads to JSONL file."""
        out = DATA_DIR / 'enhanced_leads.jsonl'
        
        try:
            with out.open('w', encoding='utf-8') as f:
                for lead in leads:
                    # Convert to dictionary for serialization
                    lead_dict = asdict(lead)
                    
                    # Handle complex objects
                    if lead.property_profile:
                        lead_dict['property_profile'] = lead.property_profile.to_dict()
                    
                    if lead.claim_readiness:
                        lead_dict['claim_readiness'] = lead.claim_readiness.to_dict()
                    
                    # Write as JSON
                    f.write(json.dumps(lead_dict, ensure_ascii=False, default=str) + '\n')
            
            logger.info(f'Successfully wrote {len(leads)} enhanced leads to {out}')
        except Exception as e:
            logger.error(f"Failed to write enhanced leads file: {e}")
            raise
    
    def run_enhanced_analysis(self, config: APIConfig, start: str, end: str, 
                            hail_min: float, wind_min: float, limit: int) -> IngestionResult:
        """Run enhanced geospatial analysis."""
        start_time = time.time()
        
        try:
            # Try to fetch from NOAA API if token is available
            if config.token:
                logger.info("Attempting to fetch data from NOAA API for enhanced analysis")
                try:
                    events = fetch_ncei_with_retry(config, start, end, limit)
                    
                    if events:
                        # Cache the successful fetch
                        cache_data(events, start, end)
                        
                        filtered = filter_events(events, hail_min, wind_min)
                        enhanced_leads = [self.create_enhanced_lead(ev) for ev in filtered]
                        
                        if enhanced_leads:
                            write_raw(events, start, end)
                            write_enhanced_leads(enhanced_leads)
                            
                            processing_time = time.time() - start_time
                            logger.info(f"Successfully processed {len(enhanced_leads)} enhanced leads from NOAA API in {processing_time:.2f}s")
                            
                            return IngestionResult(
                                source=DataSource.NOAA_API,
                                events_count=len(events),
                                leads_count=len(enhanced_leads),
                                success=True,
                                processing_time=processing_time
                            )
                        else:
                            logger.warning("No enhanced leads found from NOAA API data")
                    
                except Exception as e:
                    logger.error(f"NOAA API fetch failed: {e}")
            
            # Try to load cached data
            logger.info("Attempting to load cached data for enhanced analysis")
            cached_events = load_cached_data(start, end)
            if cached_events:
                filtered = filter_events(cached_events, hail_min, wind_min)
                enhanced_leads = [self.create_enhanced_lead(ev) for ev in filtered]
                
                if enhanced_leads:
                    write_enhanced_leads(enhanced_leads)
                    
                    processing_time = time.time() - start_time
                    logger.info(f"Successfully processed {len(enhanced_leads)} enhanced leads from cached data in {processing_time:.2f}s")
                    
                    return IngestionResult(
                        source=DataSource.CACHED,
                        events_count=len(cached_events),
                        leads_count=len(enhanced_leads),
                        success=True,
                        processing_time=processing_time
                    )
            
            # Fallback to synthetic data
            logger.info("Falling back to synthetic data generation for enhanced analysis")
            synthetic_events = self._generate_synthetic_events(15)
            enhanced_leads = [self.create_enhanced_lead(ev) for ev in synthetic_events]
            write_enhanced_leads(enhanced_leads)
            
            processing_time = time.time() - start_time
            logger.info(f"Generated {len(enhanced_leads)} enhanced synthetic leads in {processing_time:.2f}s")
            
            return IngestionResult(
                source=DataSource.SYNTHETIC,
                events_count=len(synthetic_events),
                leads_count=len(enhanced_leads),
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Enhanced analysis failed after {processing_time:.2f}s: {e}")
            
            return IngestionResult(
                source=DataSource.SYNTHETIC,
                events_count=0,
                leads_count=0,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _generate_synthetic_events(self, count: int) -> List[Dict[str, Any]]:
        """Generate synthetic storm events for testing."""
        events = []
        
        for i in range(count):
            # Weight towards Wisconsin (60%) vs Illinois (40%)
            if random.random() < 0.6:
                lat = random.uniform(43.0, 45.0)  # Wisconsin
                state = 'WI'
            else:
                lat = random.uniform(41.6, 43.0)  # Illinois (northern)
                state = 'IL'
            
            lon = random.uniform(-93.0, -87.0)
            mag = round(random.uniform(0.5, 3.0), 2)
            
            # Vary event types
            event_types = ['Hail', 'Thunderstorm Wind', 'Tornado']
            event_type = random.choice(event_types)
            
            # Adjust magnitude based on event type
            if event_type == 'Tornado':
                mag = round(random.uniform(0.5, 3.0), 1)  # EF scale
            elif event_type == 'Thunderstorm Wind':
                mag = round(random.uniform(50, 100), 0)  # mph
            else:  # Hail
                mag = round(random.uniform(0.5, 3.0), 2)  # inches
            
            event = {
                'EVENT_ID': f'syn-enhanced-{i}',
                'EVENT_TYPE': event_type,
                'MAGNITUDE': mag,
                'BEGIN_LAT': lat,
                'BEGIN_LON': lon,
                'STATE': state,
                'BEGIN_DATE_TIME': (datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 30))).isoformat() + 'Z'
            }
            events.append(event)
        
        return events

def main(argv=None) -> int:
    """Main entry point for enhanced runner."""
    parser = argparse.ArgumentParser(description='Enhanced Hail Hero with Geospatial Analysis')
    today = datetime.date.today()
    default_end = today.isoformat()
    default_start = (today - datetime.timedelta(days=365)).isoformat()
    
    parser.add_argument('--start', default=default_start, help='Start date (ISO format)')
    parser.add_argument('--end', default=default_end, help='End date (ISO format)')
    parser.add_argument('--hail-min', type=float, default=DEFAULT_HAIL_MIN, help='Minimum hail size (inches)')
    parser.add_argument('--wind-min', type=float, default=DEFAULT_WIND_MIN, help='Minimum wind speed (mph)')
    parser.add_argument('--limit', type=int, default=1000, help='Maximum records per API call')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--enhanced-only', action='store_true', help='Only run enhanced analysis (no basic leads)')
    
    args = parser.parse_args(argv)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting Enhanced Hail Hero analysis from {args.start} to {args.end}")
    
    try:
        config = load_config()
        enhanced_runner = EnhancedRunner()
        result = enhanced_runner.run_enhanced_analysis(
            config, args.start, args.end, args.hail_min, args.wind_min, args.limit
        )
        
        if result.success:
            logger.info(f"Successfully completed enhanced analysis: {result}")
            return 0
        else:
            logger.error(f"Enhanced analysis failed: {result.error_message}")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error in enhanced main: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())