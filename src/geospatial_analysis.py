#!/usr/bin/env python3
"""
Hail Hero Geospatial Analysis System

Provides advanced geospatial event correlation, claim readiness scoring,
and measurement tools for field representatives.

Core Features:
- Geospatial event correlation and clustering
- Advanced claim readiness scoring algorithms
- Property damage assessment and measurement tools
- Geographic optimization for field teams
- Temporal pattern analysis
- Risk assessment and insurance claim prediction
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'specs' / '001-hail-hero-hail' / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Constants
EARTH_RADIUS_KM = 6371.0
HAILEFFECTIVE_RADIUS_KM = 2.0  # Effective radius for hail impact
WIND_EFFECTIVE_RADIUS_KM = 5.0  # Effective radius for wind damage
CLAIM_SUCCESS_THRESHOLD = 0.65  # Minimum confidence for claim success

class EventType(Enum):
    """Storm event types."""
    HAIL = "hail"
    WIND = "wind"
    TORNADO = "tornado"
    THUNDERSTORM = "thunderstorm"
    UNKNOWN = "unknown"

class PropertyType(Enum):
    """Property types for damage assessment."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    MIXED_USE = "mixed_use"
    UNKNOWN = "unknown"

class RoofType(Enum):
    """Roof types for damage assessment."""
    ASPHALT_SHINGLES = "asphalt_shingles"
    METAL = "metal"
    TILE = "tile"
    WOOD = "wood"
    FLAT = "flat"
    UNKNOWN = "unknown"

class ClaimTier(Enum):
    """Claim readiness tiers."""
    HOT = "hot"      # 80-100: Immediate action required
    WARM = "warm"    # 60-79: High priority
    COOL = "cool"    # 40-59: Moderate priority
    COLD = "cold"    # 0-39: Low priority

@dataclass
class GeospatialBounds:
    """Geographic bounds for clustering."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    
    def contains(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within bounds."""
        return (self.min_lat <= lat <= self.max_lat and 
                self.min_lon <= lon <= self.max_lon)
    
    def center(self) -> Tuple[float, float]:
        """Get center point of bounds."""
        return ((self.min_lat + self.max_lat) / 2, 
                (self.min_lon + self.max_lon) / 2)

@dataclass
class StormEvent:
    """Individual storm event data."""
    event_id: str
    event_type: EventType
    magnitude: float
    latitude: float
    longitude: float
    timestamp: datetime.datetime
    state: str
    duration_minutes: Optional[int] = None
    affected_area_km2: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class PropertyProfile:
    """Property profile for damage assessment."""
    property_id: str
    latitude: float
    longitude: float
    property_type: PropertyType
    roof_type: RoofType
    roof_age_years: int
    property_value_usd: float
    square_footage: float
    year_built: int
    address: Optional[str] = None
    owner_name: Optional[str] = None
    contact_info: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['property_type'] = self.property_type.value
        result['roof_type'] = self.roof_type.value
        return result

@dataclass
class DamageAssessment:
    """Damage assessment results."""
    property_id: str
    estimated_damage_usd: float
    damage_percentage: float
    severity_level: str  # "minimal", "minor", "moderate", "severe", "catastrophic"
    repair_priority: int  # 1-5, where 5 is highest priority
    affected_components: List[str]
    estimated_repair_days: int
    insurance_likelihood: float  # 0.0-1.0
    assessment_date: datetime.datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['assessment_date'] = self.assessment_date.isoformat()
        return result

@dataclass
class ClaimReadinessScore:
    """Comprehensive claim readiness score."""
    property_id: str
    overall_score: float  # 0-100
    claim_tier: ClaimTier
    confidence: float  # 0.0-1.0
    component_scores: Dict[str, float]
    damage_assessment: DamageAssessment
    risk_factors: List[str]
    recommendations: List[str]
    estimated_claim_value: float
    success_probability: float
    optimal_outreach_window: Dict[str, datetime.datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['claim_tier'] = self.claim_tier.value
        result['damage_assessment'] = self.damage_assessment.to_dict()
        result['optimal_outreach_window'] = {
            'start': self.optimal_outreach_window['start'].isoformat(),
            'end': self.optimal_outreach_window['end'].isoformat()
        }
        return result

class GeospatialAnalyzer:
    """Core geospatial analysis engine."""
    
    def __init__(self):
        self.events: List[StormEvent] = []
        self.properties: List[PropertyProfile] = []
        self.claim_scores: List[ClaimReadinessScore] = []
        
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return EARTH_RADIUS_KM * c
    
    def calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(y, x)
        return math.degrees(bearing) % 360
    
    def point_in_polygon(self, lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        x, y = lon, lat
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def cluster_events_by_proximity(self, events: List[StormEvent], max_distance_km: float = 10.0) -> List[List[StormEvent]]:
        """Cluster events by geographic proximity."""
        if not events:
            return []
        
        clusters = []
        used_events = set()
        
        for event in events:
            if event.event_id in used_events:
                continue
            
            cluster = [event]
            used_events.add(event.event_id)
            
            # Find all events within max_distance_km
            for other_event in events:
                if other_event.event_id in used_events:
                    continue
                
                distance = self.haversine_distance(
                    event.latitude, event.longitude,
                    other_event.latitude, other_event.longitude
                )
                
                if distance <= max_distance_km:
                    cluster.append(other_event)
                    used_events.add(other_event.event_id)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def correlate_events_to_properties(self, events: List[StormEvent], properties: List[PropertyProfile]) -> Dict[str, List[StormEvent]]:
        """Correlate events to properties based on geographic proximity."""
        correlation = {prop.property_id: [] for prop in properties}
        
        for prop in properties:
            for event in events:
                distance = self.haversine_distance(
                    prop.latitude, prop.longitude,
                    event.latitude, event.longitude
                )
                
                # Check if property is within effective radius of event
                effective_radius = HAILEFFECTIVE_RADIUS_KM if event.event_type == EventType.HAIL else WIND_EFFECTIVE_RADIUS_KM
                
                if distance <= effective_radius:
                    correlation[prop.property_id].append(event)
        
        return correlation
    
    def calculate_event_density(self, events: List[StormEvent], radius_km: float = 25.0) -> Dict[str, float]:
        """Calculate event density for each event location."""
        density_map = {}
        
        for event in events:
            nearby_events = 0
            for other_event in events:
                if event.event_id == other_event.event_id:
                    continue
                
                distance = self.haversine_distance(
                    event.latitude, event.longitude,
                    other_event.latitude, other_event.longitude
                )
                
                if distance <= radius_km:
                    nearby_events += 1
            
            # Density as events per 1000 km²
            area_km2 = math.pi * radius_km ** 2
            density = (nearby_events / area_km2) * 1000
            density_map[event.event_id] = density
        
        return density_map

class DamageAssessmentEngine:
    """Advanced damage assessment and calculation engine."""
    
    def __init__(self):
        self.damage_multipliers = {
            RoofType.ASPHALT_SHINGLES: 1.0,
            RoofType.METAL: 0.7,
            RoofType.TILE: 1.3,
            RoofType.WOOD: 1.1,
            RoofType.FLAT: 1.2,
            RoofType.UNKNOWN: 1.0
        }
        
        self.age_degradation = {
            0-5: 1.0,      # New roof, full protection
            6-10: 0.9,    # Slight degradation
            11-15: 0.8,   # Moderate degradation
            16-20: 0.7,   # Significant degradation
            21-25: 0.6,   # Heavy degradation
            26-30: 0.5,   # Very heavy degradation
            31+: 0.4      # End of life
        }
    
    def calculate_hail_damage(self, hail_size_inches: float, roof_type: RoofType, roof_age: int) -> float:
        """Calculate damage percentage from hail."""
        # Base damage from hail size (exponential scaling)
        base_damage = min(0.95, (hail_size_inches / 4.0) ** 2)
        
        # Apply roof type multiplier
        roof_multiplier = self.damage_multipliers.get(roof_type, 1.0)
        
        # Apply age degradation
        age_key = next((k for k in self.age_degradation.keys() if roof_age in k), 31+)
        age_multiplier = self.age_degradation[age_key]
        
        # Calculate final damage
        damage_percentage = base_damage * roof_multiplier * (2 - age_multiplier)
        
        return min(0.95, damage_percentage)
    
    def calculate_wind_damage(self, wind_speed_mph: float, roof_type: RoofType, roof_age: int) -> float:
        """Calculate damage percentage from wind."""
        # Base damage from wind speed (logarithmic scaling)
        if wind_speed_mph < 60:
            base_damage = 0.0
        elif wind_speed_mph < 80:
            base_damage = (wind_speed_mph - 60) / 100
        elif wind_speed_mph < 100:
            base_damage = 0.2 + (wind_speed_mph - 80) / 50
        else:
            base_damage = min(0.95, 0.6 + (wind_speed_mph - 100) / 25)
        
        # Apply roof type multiplier
        roof_multiplier = self.damage_multipliers.get(roof_type, 1.0)
        
        # Apply age degradation
        age_key = next((k for k in self.age_degradation.keys() if roof_age in k), 31+)
        age_multiplier = self.age_degradation[age_key]
        
        # Calculate final damage
        damage_percentage = base_damage * roof_multiplier * (2 - age_multiplier)
        
        return min(0.95, damage_percentage)
    
    def assess_property_damage(self, property_profile: PropertyProfile, events: List[StormEvent]) -> DamageAssessment:
        """Assess damage to a property from multiple events."""
        total_damage_percentage = 0.0
        affected_components = []
        
        for event in events:
            if event.event_type == EventType.HAIL:
                damage = self.calculate_hail_damage(event.magnitude, property_profile.roof_type, property_profile.roof_age)
                affected_components.extend(["roof", "gutters", "siding", "windows"])
            elif event.event_type == EventType.WIND:
                damage = self.calculate_wind_damage(event.magnitude, property_profile.roof_type, property_profile.roof_age)
                affected_components.extend(["roof", "siding", "fencing", "landscaping"])
            elif event.event_type == EventType.TORNADO:
                # Tornado damage is more severe and widespread
                damage = min(0.95, event.magnitude * 0.3)  # EF scale: 0.3 per EF rating
                affected_components.extend(["roof", "structure", "windows", "doors", "landscaping"])
            else:
                damage = 0.0
            
            total_damage_percentage = min(0.95, total_damage_percentage + damage * 0.7)  # Cumulative with diminishing returns
        
        # Remove duplicates from affected components
        affected_components = list(set(affected_components))
        
        # Calculate estimated damage cost
        estimated_damage_usd = property_profile.property_value_usd * total_damage_percentage
        
        # Determine severity level
        if total_damage_percentage >= 0.75:
            severity_level = "catastrophic"
            repair_priority = 5
            estimated_repair_days = 60
        elif total_damage_percentage >= 0.50:
            severity_level = "severe"
            repair_priority = 4
            estimated_repair_days = 30
        elif total_damage_percentage >= 0.25:
            severity_level = "moderate"
            repair_priority = 3
            estimated_repair_days = 14
        elif total_damage_percentage >= 0.10:
            severity_level = "minor"
            repair_priority = 2
            estimated_repair_days = 7
        else:
            severity_level = "minimal"
            repair_priority = 1
            estimated_repair_days = 3
        
        # Calculate insurance likelihood based on damage severity and property characteristics
        insurance_likelihood = min(0.95, total_damage_percentage * 1.5)
        
        # Adjust for property type
        if property_profile.property_type == PropertyType.COMMERCIAL:
            insurance_likelihood *= 0.9  # Commercial claims more complex
        elif property_profile.property_type == PropertyType.INDUSTRIAL:
            insurance_likelihood *= 0.85  # Industrial claims very complex
        
        return DamageAssessment(
            property_id=property_profile.property_id,
            estimated_damage_usd=estimated_damage_usd,
            damage_percentage=total_damage_percentage,
            severity_level=severity_level,
            repair_priority=repair_priority,
            affected_components=affected_components,
            estimated_repair_days=estimated_repair_days,
            insurance_likelihood=insurance_likelihood,
            assessment_date=datetime.datetime.utcnow()
        )

class ClaimScoringEngine:
    """Advanced claim readiness scoring engine."""
    
    def __init__(self):
        self.geospatial_analyzer = GeospatialAnalyzer()
        self.damage_engine = DamageAssessmentEngine()
    
    def calculate_claim_readiness_score(self, property_profile: PropertyProfile, events: List[StormEvent]) -> ClaimReadinessScore:
        """Calculate comprehensive claim readiness score."""
        
        # Assess damage
        damage_assessment = self.damage_engine.assess_property_damage(property_profile, events)
        
        # Calculate component scores
        component_scores = {}
        
        # 1. Damage Severity Score (0-100)
        damage_score = min(100, damage_assessment.damage_percentage * 100)
        component_scores['damage_severity'] = damage_score
        
        # 2. Event Frequency Score (0-100)
        event_frequency_score = min(100, len(events) * 25)
        component_scores['event_frequency'] = event_frequency_score
        
        # 3. Property Value Score (0-100)
        property_value_score = min(100, math.log10(property_profile.property_value_usd + 1) * 10)
        component_scores['property_value'] = property_value_score
        
        # 4. Insurance Likelihood Score (0-100)
        insurance_score = damage_assessment.insurance_likelihood * 100
        component_scores['insurance_likelihood'] = insurance_score
        
        # 5. Urgency Score (0-100)
        urgency_score = damage_assessment.repair_priority * 20
        component_scores['urgency'] = urgency_score
        
        # 6. Seasonal Timing Score (0-100)
        current_month = datetime.datetime.utcnow().month
        if 5 <= current_month <= 8:  # Peak hail season
            seasonal_score = 90
        elif 3 <= current_month <= 4 or 9 <= current_month <= 10:  # Shoulder seasons
            seasonal_score = 70
        else:  # Off-season
            seasonal_score = 50
        component_scores['seasonal_timing'] = seasonal_score
        
        # Calculate weighted overall score
        weights = {
            'damage_severity': 0.30,
            'event_frequency': 0.20,
            'property_value': 0.15,
            'insurance_likelihood': 0.15,
            'urgency': 0.10,
            'seasonal_timing': 0.10
        }
        
        overall_score = sum(component_scores[component] * weights[component] for component in weights)
        
        # Determine claim tier
        if overall_score >= 80:
            claim_tier = ClaimTier.HOT
        elif overall_score >= 60:
            claim_tier = ClaimTier.WARM
        elif overall_score >= 40:
            claim_tier = ClaimTier.COOL
        else:
            claim_tier = ClaimTier.COLD
        
        # Calculate confidence based on data quality
        confidence_factors = []
        if property_profile.property_type != PropertyType.UNKNOWN:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.7)
        
        if property_profile.roof_type != RoofType.UNKNOWN:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.7)
        
        if property_profile.property_value_usd > 0:
            confidence_factors.append(0.95)
        else:
            confidence_factors.append(0.6)
        
        confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Generate risk factors
        risk_factors = []
        if damage_assessment.damage_percentage < 0.10:
            risk_factors.append("Minimal damage - may not meet deductible")
        if property_profile.roof_age > 20:
            risk_factors.append("Old roof - depreciation may reduce claim value")
        if len(events) == 0:
            risk_factors.append("No recent storm events - claim may be questioned")
        if property_profile.property_type == PropertyType.COMMERCIAL:
            risk_factors.append("Commercial property - complex claim process")
        
        # Generate recommendations
        recommendations = []
        if damage_assessment.severity_level in ["severe", "catastrophic"]:
            recommendations.append("Immediate inspection required")
            recommendations.append("Contact insurance company ASAP")
        
        if damage_assessment.insurance_likelihood > 0.7:
            recommendations.append("High likelihood of successful claim")
            recommendations.append("Document all damage with photos")
        
        if property_profile.roof_age > 15:
            recommendations.append("Consider roof replacement vs repair")
        
        # Calculate optimal outreach window
        current_time = datetime.datetime.utcnow()
        optimal_start = current_time + datetime.timedelta(days=1)
        optimal_end = current_time + datetime.timedelta(days=7)
        
        # Adjust for severity
        if damage_assessment.severity_level in ["severe", "catastrophic"]:
            optimal_start = current_time
            optimal_end = current_time + datetime.timedelta(days=3)
        
        optimal_outreach_window = {
            'start': optimal_start,
            'end': optimal_end
        }
        
        # Calculate success probability
        success_probability = damage_assessment.insurance_likelihood * confidence
        
        # Calculate estimated claim value
        estimated_claim_value = damage_assessment.estimated_damage_usd * 0.8  # 80% of damage value typical
        
        return ClaimReadinessScore(
            property_id=property_profile.property_id,
            overall_score=overall_score,
            claim_tier=claim_tier,
            confidence=confidence,
            component_scores=component_scores,
            damage_assessment=damage_assessment,
            risk_factors=risk_factors,
            recommendations=recommendations,
            estimated_claim_value=estimated_claim_value,
            success_probability=success_probability,
            optimal_outreach_window=optimal_outreach_window
        )

class FieldToolsIntegration:
    """Integration tools for field representatives."""
    
    def __init__(self):
        self.geospatial_analyzer = GeospatialAnalyzer()
    
    def create_inspection_workflow(self, property_profile: PropertyProfile, claim_score: ClaimReadinessScore) -> Dict[str, Any]:
        """Create comprehensive inspection workflow for field representatives."""
        
        workflow = {
            'property_id': property_profile.property_id,
            'property_details': property_profile.to_dict(),
            'claim_score': claim_score.to_dict(),
            'inspection_checklist': self._generate_inspection_checklist(claim_score),
            'measurement_requirements': self._generate_measurement_requirements(claim_score),
            'photo_requirements': self._generate_photo_requirements(claim_score),
            'safety_considerations': self._generate_safety_considerations(claim_score),
            'documentation_requirements': self._generate_documentation_requirements(claim_score),
            'communication_template': self._generate_communication_template(claim_score)
        }
        
        return workflow
    
    def _generate_inspection_checklist(self, claim_score: ClaimReadinessScore) -> List[Dict[str, Any]]:
        """Generate detailed inspection checklist."""
        checklist = []
        
        # Roof inspection
        checklist.append({
            'category': 'Roof Inspection',
            'items': [
                'Check for missing, cracked, or curled shingles',
                'Inspect for hail damage (dents, bruising)',
                'Check flashing around penetrations',
                'Inspect gutters and downspouts',
                'Check for granule loss in shingles',
                'Look for water stains on ceiling'
            ],
            'priority': 'High' if claim_score.damage_assessment.damage_percentage > 0.25 else 'Medium'
        })
        
        # Exterior inspection
        checklist.append({
            'category': 'Exterior Inspection',
            'items': [
                'Check siding for dents and damage',
                'Inspect windows and frames',
                'Check garage door for damage',
                'Inspect fencing and landscaping',
                'Look for debris impact damage'
            ],
            'priority': 'Medium'
        })
        
        # Interior inspection
        if claim_score.damage_assessment.damage_percentage > 0.30:
            checklist.append({
                'category': 'Interior Inspection',
                'items': [
                    'Check ceilings for water stains',
                    'Inspect walls for cracks or water damage',
                    'Check windows and doors for proper operation',
                    'Look for signs of water intrusion'
                ],
                'priority': 'High'
            })
        
        return checklist
    
    def _generate_measurement_requirements(self, claim_score: ClaimReadinessScore) -> List[Dict[str, Any]]:
        """Generate measurement requirements for damage assessment."""
        measurements = []
        
        if claim_score.damage_assessment.damage_percentage > 0.10:
            measurements.append({
                'type': 'Roof Measurements',
                'required': True,
                'details': [
                    'Measure total roof area',
                    'Measure damaged areas separately',
                    'Calculate percentage of damage',
                    'Note pitch and complexity factors'
                ]
            })
        
        if claim_score.damage_assessment.damage_percentage > 0.20:
            measurements.append({
                'type': 'Siding Measurements',
                'required': True,
                'details': [
                    'Measure total siding area',
                    'Measure damaged sections',
                    'Identify siding material and type'
                ]
            })
        
        return measurements
    
    def _generate_photo_requirements(self, claim_score: ClaimReadinessScore) -> List[Dict[str, Any]]:
        """Generate photo documentation requirements."""
        photos = []
        
        # Required photos for all claims
        photos.append({
            'category': 'Property Overview',
            'required': True,
            'shots': [
                'Front of property (distance shot)',
                'Back of property',
                'Street view showing context',
                'Property address/identification'
            ]
        })
        
        # Damage-specific photos
        if claim_score.damage_assessment.damage_percentage > 0.05:
            photos.append({
                'category': 'Damage Documentation',
                'required': True,
                'shots': [
                    'Close-up of roof damage',
                    'Medium shots of damaged areas',
                    'Photos with measurement reference',
                    'Photos showing different angles'
                ]
            })
        
        # Detail shots for significant damage
        if claim_score.damage_assessment.damage_percentage > 0.25:
            photos.append({
                'category': 'Detailed Documentation',
                'required': True,
                'shots': [
                    'Individual shingle damage',
                    'Gutter and downspout damage',
                    'Window and door damage',
                    'Interior damage if present'
                ]
            })
        
        return photos
    
    def _generate_safety_considerations(self, claim_score: ClaimReadinessScore) -> List[str]:
        """Generate safety considerations for field representatives."""
        safety = []
        
        # General safety
        safety.extend([
            'Wear appropriate PPE (hard hat, gloves, safety glasses)',
            'Use fall protection when accessing roof',
            'Check weather conditions before inspection',
            'Be aware of electrical hazards'
        ])
        
        # Damage-specific safety
        if claim_score.damage_assessment.damage_percentage > 0.50:
            safety.extend([
                'Structural integrity may be compromised',
                'Avoid areas with severe structural damage',
                'Use extreme caution on damaged roofs',
                'Consider professional engineering assessment'
            ])
        
        return safety
    
    def _generate_documentation_requirements(self, claim_score: ClaimReadinessScore) -> List[Dict[str, Any]]:
        """Generate documentation requirements."""
        docs = []
        
        docs.append({
            'type': 'Property Information',
            'required': True,
            'items': [
                'Property owner information',
                'Insurance policy details',
                'Property age and condition',
                'Previous damage history'
            ]
        })
        
        docs.append({
            'type': 'Damage Report',
            'required': True,
            'items': [
                'Detailed damage description',
                'Measurement calculations',
                'Photo documentation',
                'Repair cost estimates',
                'Recommended actions'
            ]
        })
        
        return docs
    
    def _generate_communication_template(self, claim_score: ClaimReadinessScore) -> Dict[str, str]:
        """Generate communication templates for property owners."""
        
        if claim_score.overall_score >= 60:
            subject = "Important: Storm Damage Assessment for Your Property"
            body = f"""Dear Property Owner,

Our storm damage assessment indicates significant damage to your property that may be covered by your insurance policy.

Assessment Summary:
- Damage Level: {claim_score.damage_assessment.severity_level.title()}
- Estimated Damage: ${claim_score.damage_assessment.estimated_damage_usd:,.2f}
- Claim Success Probability: {claim_score.success_probability:.1%}

We recommend immediate action to document this damage and begin the insurance claim process.

Please contact us to schedule a detailed inspection.

Best regards,
Hail Hero Team
"""
        else:
            subject = "Storm Damage Assessment Results"
            body = f"""Dear Property Owner,

We have completed a storm damage assessment for your property.

Assessment Summary:
- Damage Level: {claim_score.damage_assessment.severity_level.title()}
- Estimated Damage: ${claim_score.damage_assessment.estimated_damage_usd:,.2f}
- Claim Success Probability: {claim_score.success_probability:.1%}

{self._generate_recommendations_text(claim_score)}

If you have any questions or would like to discuss this further, please contact us.

Best regards,
Hail Hero Team
"""
        
        return {
            'email_subject': subject,
            'email_body': body,
            'sms_summary': f"Storm damage assessment complete. {claim_score.damage_assessment.severity_level.title()} damage detected. Contact us for details."
        }
    
    def _generate_recommendations_text(self, claim_score: ClaimReadinessScore) -> str:
        """Generate recommendations text based on claim score."""
        if claim_score.overall_score >= 80:
            return "Immediate action is recommended. Contact your insurance company and schedule a professional inspection."
        elif claim_score.overall_score >= 60:
            return "We recommend filing an insurance claim and scheduling repairs soon."
        elif claim_score.overall_score >= 40:
            return "Consider monitoring the damage and consulting with a professional if it worsens."
        else:
            return "Damage appears minimal. Continue to monitor and contact us if you notice any changes."

# Initialize the main system
geospatial_system = GeospatialAnalyzer()
damage_engine = DamageAssessmentEngine()
claim_scoring_engine = ClaimScoringEngine()
field_tools = FieldToolsIntegration()

if __name__ == "__main__":
    print("Hail Hero Geospatial Analysis System initialized")
    print("Ready for storm event analysis and claim readiness scoring")