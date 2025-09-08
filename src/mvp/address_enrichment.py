#!/usr/bin/env python3
"""Address enrichment module for Hail Hero MVP.

This module provides comprehensive address enrichment functionality including:
- OpenAddresses integration for property data
- OpenStreetMap/Nominatim integration for geocoding
- Address normalization and standardization
- Property attribute enrichment
- Geographic information retrieval
- Data quality scoring
- Property deduplication
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import quote, urlencode
import hashlib
import math

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AddressEnricher:
    """Main address enrichment class."""
    
    def __init__(self, nominatim_base_url: str = "https://nominatim.openstreetmap.org",
                 openaddresses_base_url: str = "https://openaddresses.io",
                 user_agent: str = "HailHero-MVP/1.0"):
        self.nominatim_base_url = nominatim_base_url
        self.openaddresses_base_url = openaddresses_base_url
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
        
        # Cache for geocoding results
        self.geocode_cache: Dict[str, Dict[str, Any]] = {}
        
        # Property deduplication cache
        self.property_cache: Dict[str, Dict[str, Any]] = {}
        
    def normalize_address(self, address: str) -> Dict[str, str]:
        """Normalize and standardize address components."""
        if not address:
            return {}
            
        # Convert to uppercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', address.strip().upper())
        
        # Extract common components
        components = {
            'street_number': '',
            'street_name': '',
            'street_type': '',
            'city': '',
            'state': '',
            'zip_code': '',
            'country': 'USA'
        }
        
        # Simple regex patterns for US addresses
        street_num_pattern = r'^(\d+)\s+'
        street_type_pattern = r'\b(ST|STREET|AVE|AVENUE|BLVD|BOULEVARD|DR|DRIVE|RD|ROAD|LN|LANE|CT|COURT|PL|PLACE|WAY)\b'
        zip_pattern = r'\b(\d{5})(?:-\d{4})?\b'
        state_pattern = r'\b([A-Z]{2})\b'
        
        # Extract street number
        street_num_match = re.search(street_num_pattern, normalized)
        if street_num_match:
            components['street_number'] = street_num_match.group(1)
            normalized = normalized[street_num_match.end():]
        
        # Extract street type
        street_type_match = re.search(street_type_pattern, normalized)
        if street_type_match:
            components['street_type'] = street_type_match.group(1)
            normalized = normalized[:street_type_match.start()] + normalized[street_type_match.end():]
        
        # Extract zip code
        zip_match = re.search(zip_pattern, normalized)
        if zip_match:
            components['zip_code'] = zip_match.group(1)
            normalized = normalized[:zip_match.start()] + normalized[zip_match.end():]
        
        # Extract state
        state_match = re.search(state_pattern, normalized)
        if state_match:
            components['state'] = state_match.group(1)
            normalized = normalized[:state_match.start()] + normalized[state_match.end():]
        
        # Remaining text is likely street name and city
        parts = normalized.split(',')
        if len(parts) > 1:
            components['street_name'] = parts[0].strip()
            components['city'] = parts[-1].strip()
        else:
            components['street_name'] = parts[0].strip()
        
        return components
    
    def geocode_address(self, address: str, lat: Optional[float] = None, lon: Optional[float] = None) -> Dict[str, Any]:
        """Geocode address using Nominatim with reverse geocoding fallback."""
        cache_key = f"{address}_{lat}_{lon}"
        
        if cache_key in self.geocode_cache:
            return self.geocode_cache[cache_key]
        
        result = {
            'address': address,
            'latitude': lat,
            'longitude': lon,
            'normalized_address': '',
            'components': {},
            'quality_score': 0,
            'geocoding_method': 'none'
        }
        
        try:
            # Try to geocode the address
            if address:
                geocode_result = self._forward_geocode(address)
                if geocode_result:
                    result.update(geocode_result)
                    result['geocoding_method'] = 'forward'
            
            # If no result from address geocoding, try reverse geocoding
            if not result['normalized_address'] and lat is not None and lon is not None:
                reverse_result = self._reverse_geocode(lat, lon)
                if reverse_result:
                    result.update(reverse_result)
                    result['geocoding_method'] = 'reverse'
            
            # Normalize the address components
            if result.get('normalized_address'):
                result['components'] = self.normalize_address(result['normalized_address'])
            
            # Calculate quality score
            result['quality_score'] = self._calculate_geocoding_quality(result)
            
        except Exception as e:
            logger.error(f"Geocoding error for {address}: {e}")
        
        # Cache the result
        self.geocode_cache[cache_key] = result
        return result
    
    def _forward_geocode(self, address: str) -> Optional[Dict[str, Any]]:
        """Forward geocode using Nominatim."""
        try:
            params = {
                'q': address,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1,
                'countrycodes': 'us'
            }
            
            url = f"{self.nominatim_base_url}/search"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data:
                result = data[0]
                return {
                    'normalized_address': result.get('display_name', ''),
                    'latitude': float(result.get('lat', 0)),
                    'longitude': float(result.get('lon', 0)),
                    'address_type': result.get('type', ''),
                    'importance': result.get('importance', 0),
                    'osm_type': result.get('osm_type', ''),
                    'osm_id': result.get('osm_id', ''),
                    'raw_geocode_data': result
                }
        
        except Exception as e:
            logger.error(f"Forward geocoding error: {e}")
        
        return None
    
    def _reverse_geocode(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Reverse geocode using Nominatim."""
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'addressdetails': 1,
                'zoom': 18
            }
            
            url = f"{self.nominatim_base_url}/reverse"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data:
                return {
                    'normalized_address': data.get('display_name', ''),
                    'latitude': lat,
                    'longitude': lon,
                    'address_type': data.get('type', ''),
                    'importance': data.get('importance', 0),
                    'osm_type': data.get('osm_type', ''),
                    'osm_id': data.get('osm_id', ''),
                    'raw_geocode_data': data
                }
        
        except Exception as e:
            logger.error(f"Reverse geocoding error: {e}")
        
        return None
    
    def _calculate_geocoding_quality(self, geocode_result: Dict[str, Any]) -> float:
        """Calculate quality score for geocoding result."""
        score = 0.0
        
        # Base score for having coordinates
        if geocode_result.get('latitude') and geocode_result.get('longitude'):
            score += 30
        
        # Score for normalized address
        if geocode_result.get('normalized_address'):
            score += 20
        
        # Score for address components
        components = geocode_result.get('components', {})
        if components.get('street_number'):
            score += 15
        if components.get('street_name'):
            score += 15
        if components.get('city'):
            score += 10
        if components.get('state'):
            score += 5
        if components.get('zip_code'):
            score += 5
        
        # Score for importance/reliability
        importance = geocode_result.get('importance', 0)
        score += min(importance * 100, 20)
        
        return min(score, 100)
    
    def enrich_property_data(self, lat: float, lon: float, address: str = "") -> Dict[str, Any]:
        """Enrich property data with various attributes."""
        enriched_data = {
            'coordinates': {'lat': lat, 'lon': lon},
            'address': address,
            'property_attributes': {},
            'geographic_info': {},
            'assessment_data': {},
            'quality_score': 0
        }
        
        try:
            # Get geographic information
            geo_info = self._get_geographic_info(lat, lon)
            enriched_data['geographic_info'] = geo_info
            
            # Simulate property attributes (in real implementation, would integrate with property databases)
            property_attrs = self._simulate_property_attributes(lat, lon, address)
            enriched_data['property_attributes'] = property_attrs
            
            # Simulate assessment data
            assessment_data = self._simulate_assessment_data(lat, lon)
            enriched_data['assessment_data'] = assessment_data
            
            # Calculate overall quality score
            enriched_data['quality_score'] = self._calculate_property_quality_score(enriched_data)
            
        except Exception as e:
            logger.error(f"Property enrichment error: {e}")
        
        return enriched_data
    
    def _get_geographic_info(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get geographic information for coordinates."""
        geo_info = {
            'county': '',
            'state': '',
            'country': 'USA',
            'timezone': '',
            'elevation': 0,
            'nearest_city': '',
            'distance_to_city': 0,
            'census_tract': '',
            'zip_codes': []
        }
        
        try:
            # Use reverse geocoding to get administrative divisions
            reverse_result = self._reverse_geocode(lat, lon)
            if reverse_result:
                raw_data = reverse_result.get('raw_geocode_data', {})
                address_data = raw_data.get('address', {})
                
                geo_info['county'] = address_data.get('county', '')
                geo_info['state'] = address_data.get('state', '')
                geo_info['country'] = address_data.get('country', 'USA')
                geo_info['nearest_city'] = address_data.get('city', address_data.get('town', address_data.get('village', '')))
                
                # Extract zip codes
                if address_data.get('postcode'):
                    geo_info['zip_codes'] = [address_data['postcode']]
            
            # Simulate timezone based on longitude
            if lon < -87.5:  # Central Time
                geo_info['timezone'] = 'America/Chicago'
            elif lon < -82.5:  # Eastern Time
                geo_info['timezone'] = 'America/New_York'
            else:  # Pacific/Mountain
                geo_info['timezone'] = 'America/Denver'
            
            # Simulate elevation (Wisconsin/Illinois range)
            geo_info['elevation'] = random.uniform(200, 600)  # meters
            
        except Exception as e:
            logger.error(f"Geographic info error: {e}")
        
        return geo_info
    
    def _simulate_property_attributes(self, lat: float, lon: float, address: str) -> Dict[str, Any]:
        """Simulate property attributes (would integrate with real property databases)."""
        
        # Generate realistic property attributes based on location
        attributes = {
            'property_type': random.choice(['Single Family', 'Multi Family', 'Condo', 'Townhouse']),
            'year_built': random.randint(1950, 2023),
            'square_footage': random.randint(1200, 4500),
            'lot_size': random.randint(5000, 25000),  # square feet
            'bedrooms': random.randint(2, 5),
            'bathrooms': round(random.uniform(1.5, 4.5), 1),
            'stories': random.randint(1, 3),
            'garage_spaces': random.randint(0, 3),
            'has_basement': random.choice([True, False]),
            'has_pool': random.choice([True, False]),
            'roof_type': random.choice(['Asphalt Shingle', 'Metal', 'Tile', 'Flat']),
            'exterior_material': random.choice(['Vinyl', 'Brick', 'Wood', 'Stucco']),
            'foundation_type': random.choice(['Concrete', 'Crawl Space', 'Basement']),
            'heating_type': random.choice(['Forced Air', 'Boiler', 'Heat Pump']),
            'cooling_type': random.choice(['Central AC', 'Window Units', 'None']),
            'property_condition': random.choice(['Excellent', 'Good', 'Fair', 'Poor']),
            'last_sale_date': f"20{random.randint(15, 23)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            'estimated_value': random.randint(150000, 800000),
            'assessed_value': random.randint(120000, 650000)
        }
        
        # Adjust attributes based on location patterns
        if lon < -90:  # Western Wisconsin
            attributes['property_type'] = 'Single Family'
            attributes['lot_size'] = random.randint(10000, 50000)
            attributes['has_basement'] = True
        elif lat > 42.5:  # Northern Wisconsin
            attributes['square_footage'] = random.randint(1500, 3000)
            attributes['heating_type'] = 'Forced Air'
        
        return attributes
    
    def _simulate_assessment_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Simulate property assessment data."""
        
        return {
            'assessment_year': 2023,
            'land_value': random.randint(30000, 150000),
            'improvement_value': random.randint(100000, 500000),
            'total_assessed_value': random.randint(130000, 650000),
            'tax_rate': random.uniform(0.012, 0.025),  # 1.2% to 2.5%
            'annual_taxes': random.randint(2000, 12000),
            'exemptions': random.choice(['Homestead', 'Senior', 'Veteran', 'None']),
            'assessment_district': f"District-{random.randint(1, 10)}",
            'parcel_number': f"{random.randint(100000, 999999)}-{random.randint(100, 999)}",
            'zoning': random.choice(['R1', 'R2', 'R3', 'R4', 'Commercial']),
            'flood_zone': random.choice(['A', 'AE', 'X', 'D']),
            'special_districts': random.choice(['School', 'Fire', 'Library', 'Multiple'])
        }
    
    def _calculate_property_quality_score(self, enriched_data: Dict[str, Any]) -> float:
        """Calculate quality score for enriched property data."""
        score = 0.0
        
        # Score for geographic information
        geo_info = enriched_data.get('geographic_info', {})
        if geo_info.get('county'):
            score += 10
        if geo_info.get('state'):
            score += 5
        if geo_info.get('timezone'):
            score += 5
        if geo_info.get('zip_codes'):
            score += 10
        
        # Score for property attributes
        property_attrs = enriched_data.get('property_attributes', {})
        if property_attrs.get('property_type'):
            score += 15
        if property_attrs.get('square_footage'):
            score += 10
        if property_attrs.get('year_built'):
            score += 10
        if property_attrs.get('estimated_value'):
            score += 10
        
        # Score for assessment data
        assessment_data = enriched_data.get('assessment_data', {})
        if assessment_data.get('parcel_number'):
            score += 15
        if assessment_data.get('total_assessed_value'):
            score += 10
        
        return min(score, 100)
    
    def generate_property_id(self, lat: float, lon: float, address: str = "") -> str:
        """Generate unique property ID for deduplication."""
        # Create a hash based on coordinates and address
        unique_string = f"{lat:.6f}_{lon:.6f}_{address.lower().strip()}"
        return f"prop_{hashlib.md5(unique_string.encode()).hexdigest()[:12]}"
    
    def deduplicate_properties(self, properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate properties based on location and address similarity."""
        unique_properties = []
        seen_ids = set()
        
        for prop in properties:
            lat = prop.get('coordinates', {}).get('lat')
            lon = prop.get('coordinates', {}).get('lon')
            address = prop.get('address', '')
            
            if lat is not None and lon is not None:
                prop_id = self.generate_property_id(lat, lon, address)
                
                if prop_id not in seen_ids:
                    seen_ids.add(prop_id)
                    prop['property_id'] = prop_id
                    unique_properties.append(prop)
        
        return unique_properties
    
    def enrich_lead(self, lead: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a single lead with address and property data."""
        enriched_lead = lead.copy()
        
        try:
            # Extract coordinates from lead
            property_data = lead.get('property', {})
            lat = property_data.get('lat')
            lon = property_data.get('lon')
            
            if lat is not None and lon is not None:
                # Geocode the location
                geocode_result = self.geocode_address("", lat, lon)
                
                # Enrich property data
                property_enrichment = self.enrich_property_data(lat, lon, geocode_result.get('normalized_address', ''))
                
                # Update lead with enriched data
                enriched_lead['property'].update({
                    'geocode_data': geocode_result,
                    'enrichment_data': property_enrichment,
                    'property_id': self.generate_property_id(lat, lon, geocode_result.get('normalized_address', ''))
                })
                
                # Add enrichment metadata
                enriched_lead['enrichment_metadata'] = {
                    'enriched_at': time.time(),
                    'enrichment_version': '1.0',
                    'data_sources': ['nominatim', 'simulated_property_data'],
                    'quality_score': property_enrichment.get('quality_score', 0)
                }
                
                logger.info(f"Enriched lead {lead.get('lead_id')} with property data")
        
        except Exception as e:
            logger.error(f"Error enriching lead {lead.get('lead_id')}: {e}")
            enriched_lead['enrichment_error'] = str(e)
        
        return enriched_lead
    
    def enrich_leads(self, leads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich multiple leads with address and property data."""
        enriched_leads = []
        
        for i, lead in enumerate(leads):
            logger.info(f"Enriching lead {i+1}/{len(leads)}: {lead.get('lead_id')}")
            enriched_lead = self.enrich_lead(lead)
            enriched_leads.append(enriched_lead)
            
            # Rate limiting
            if i < len(leads) - 1:
                time.sleep(1)  # 1 second delay between requests
        
        return enriched_leads


def create_enricher() -> AddressEnricher:
    """Create and configure the address enricher."""
    return AddressEnricher(
        nominatim_base_url="https://nominatim.openstreetmap.org",
        user_agent="HailHero-MVP/1.0"
    )


# Utility functions
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in miles."""
    # Haversine formula
    R = 3959  # Earth's radius in miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def is_within_radius(lat1: float, lon1: float, lat2: float, lon2: float, radius_miles: float) -> bool:
    """Check if two coordinates are within a given radius."""
    distance = calculate_distance(lat1, lon1, lat2, lon2)
    return distance <= radius_miles


if __name__ == "__main__":
    # Test the enricher
    enricher = create_enricher()
    
    # Test geocoding
    test_lat = 43.0642
    test_lon = -89.4009
    result = enricher.geocode_address("", test_lat, test_lon)
    print(f"Geocoding result: {result}")
    
    # Test property enrichment
    property_data = enricher.enrich_property_data(test_lat, test_lon)
    print(f"Property enrichment: {property_data}")
    
    # Test with a sample lead
    sample_lead = {
        'lead_id': 'test-lead',
        'property': {'lat': test_lat, 'lon': test_lon}
    }
    
    enriched_lead = enricher.enrich_lead(sample_lead)
    print(f"Enriched lead: {json.dumps(enriched_lead, indent=2)}")