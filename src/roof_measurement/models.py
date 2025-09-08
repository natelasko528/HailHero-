"""
Core data models for roof measurement system
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import json


class RoofType(Enum):
    """Roof type enumeration"""
    GABLE = "gable"
    HIP = "hip"
    FLAT = "flat"
    MANSARD = "mansard"
    GAMBREL = "gambrel"
    SHED = "shed"
    BUTTERFLY = "butterfly"
    COMPLEX = "complex"


class MaterialType(Enum):
    """Roofing material types"""
    ASPHALT_SHINGLES = "asphalt_shingles"
    METAL = "metal"
    TILE = "tile"
    SLATE = "slate"
    WOOD_SHINGLES = "wood_shingles"
    RUBBER = "rubber"
    BUILT_UP = "built_up"
    MODIFIED_BITUMEN = "modified_bitumen"


class DamageSeverity(Enum):
    """Damage severity levels"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"


@dataclass
class PropertyLocation:
    """Property location information"""
    address: str
    city: str
    state: str
    zip_code: str
    latitude: float
    longitude: float
    country: str = "USA"
    elevation: Optional[float] = None


@dataclass
class RoofSegment:
    """Individual roof segment measurement"""
    segment_id: str
    roof_type: RoofType
    area_sq_ft: float
    perimeter_ft: float
    pitch_degrees: float
    pitch_ratio: float
    complexity_factor: float = 1.0
    material_type: Optional[MaterialType] = None
    orientation: Optional[str] = None
    coordinates: List[Tuple[float, float]] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoofMeasurement:
    """Complete roof measurement data"""
    measurement_id: str
    property_location: PropertyLocation
    timestamp: datetime
    total_area_sq_ft: float
    total_perimeter_ft: float
    segments: List[RoofSegment]
    average_pitch_degrees: float
    average_pitch_ratio: float
    roof_complexity: float
    estimated_height_ft: float
    number_of_stories: int
    imagery_metadata: Dict[str, Any] = field(default_factory=dict)
    measurement_accuracy: float = 0.95
    quality_score: float = 0.9
    source_imagery_url: Optional[str] = None
    validation_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'measurement_id': self.measurement_id,
            'property_location': {
                'address': self.property_location.address,
                'city': self.property_location.city,
                'state': self.property_location.state,
                'zip_code': self.property_location.zip_code,
                'latitude': self.property_location.latitude,
                'longitude': self.property_location.longitude,
                'country': self.property_location.country,
                'elevation': self.property_location.elevation
            },
            'timestamp': self.timestamp.isoformat(),
            'total_area_sq_ft': self.total_area_sq_ft,
            'total_perimeter_ft': self.total_perimeter_ft,
            'segments': [
                {
                    'segment_id': seg.segment_id,
                    'roof_type': seg.roof_type.value,
                    'area_sq_ft': seg.area_sq_ft,
                    'perimeter_ft': seg.perimeter_ft,
                    'pitch_degrees': seg.pitch_degrees,
                    'pitch_ratio': seg.pitch_ratio,
                    'complexity_factor': seg.complexity_factor,
                    'material_type': seg.material_type.value if seg.material_type else None,
                    'orientation': seg.orientation,
                    'coordinates': seg.coordinates,
                    'features': seg.features
                }
                for seg in self.segments
            ],
            'average_pitch_degrees': self.average_pitch_degrees,
            'average_pitch_ratio': self.average_pitch_ratio,
            'roof_complexity': self.roof_complexity,
            'estimated_height_ft': self.estimated_height_ft,
            'number_of_stories': self.number_of_stories,
            'imagery_metadata': self.imagery_metadata,
            'measurement_accuracy': self.measurement_accuracy,
            'quality_score': self.quality_score,
            'source_imagery_url': self.source_imagery_url,
            'validation_notes': self.validation_notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoofMeasurement':
        """Create from dictionary"""
        property_location = PropertyLocation(
            address=data['property_location']['address'],
            city=data['property_location']['city'],
            state=data['property_location']['state'],
            zip_code=data['property_location']['zip_code'],
            latitude=data['property_location']['latitude'],
            longitude=data['property_location']['longitude'],
            country=data['property_location'].get('country', 'USA'),
            elevation=data['property_location'].get('elevation')
        )
        
        segments = []
        for seg_data in data['segments']:
            segment = RoofSegment(
                segment_id=seg_data['segment_id'],
                roof_type=RoofType(seg_data['roof_type']),
                area_sq_ft=seg_data['area_sq_ft'],
                perimeter_ft=seg_data['perimeter_ft'],
                pitch_degrees=seg_data['pitch_degrees'],
                pitch_ratio=seg_data['pitch_ratio'],
                complexity_factor=seg_data.get('complexity_factor', 1.0),
                material_type=MaterialType(seg_data['material_type']) if seg_data.get('material_type') else None,
                orientation=seg_data.get('orientation'),
                coordinates=seg_data.get('coordinates', []),
                features=seg_data.get('features', {})
            )
            segments.append(segment)
        
        return cls(
            measurement_id=data['measurement_id'],
            property_location=property_location,
            timestamp=datetime.fromisoformat(data['timestamp']),
            total_area_sq_ft=data['total_area_sq_ft'],
            total_perimeter_ft=data['total_perimeter_ft'],
            segments=segments,
            average_pitch_degrees=data['average_pitch_degrees'],
            average_pitch_ratio=data['average_pitch_ratio'],
            roof_complexity=data['roof_complexity'],
            estimated_height_ft=data['estimated_height_ft'],
            number_of_stories=data['number_of_stories'],
            imagery_metadata=data.get('imagery_metadata', {}),
            measurement_accuracy=data.get('measurement_accuracy', 0.95),
            quality_score=data.get('quality_score', 0.9),
            source_imagery_url=data.get('source_imagery_url'),
            validation_notes=data.get('validation_notes', [])
        )


@dataclass
class MaterialItem:
    """Individual material item for estimation"""
    material_name: str
    unit_type: str  # "sq_ft", "linear_ft", "each", etc.
    quantity: float
    unit_cost: float
    total_cost: float
    waste_factor: float = 0.1
    category: str = "materials"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'material_name': self.material_name,
            'unit_type': self.unit_type,
            'quantity': self.quantity,
            'unit_cost': self.unit_cost,
            'total_cost': self.total_cost,
            'waste_factor': self.waste_factor,
            'category': self.category
        }


@dataclass
class MaterialEstimate:
    """Complete material and cost estimation"""
    estimate_id: str
    measurement_id: str
    roof_measurement: RoofMeasurement
    materials: List[MaterialItem]
    labor_cost: float
    total_cost: float
    cost_per_sq_ft: float
    estimate_date: datetime
    material_type: MaterialType
    regional_pricing_factor: float = 1.0
    market_conditions: str = "normal"
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'estimate_id': self.estimate_id,
            'measurement_id': self.measurement_id,
            'roof_measurement': self.roof_measurement.to_dict(),
            'materials': [mat.to_dict() for mat in self.materials],
            'labor_cost': self.labor_cost,
            'total_cost': self.total_cost,
            'cost_per_sq_ft': self.cost_per_sq_ft,
            'estimate_date': self.estimate_date.isoformat(),
            'material_type': self.material_type.value,
            'regional_pricing_factor': self.regional_pricing_factor,
            'market_conditions': self.market_conditions,
            'notes': self.notes
        }


@dataclass
class DamageArea:
    """Individual damage area"""
    damage_id: str
    segment_id: str
    damage_type: str
    severity: DamageSeverity
    area_sq_ft: float
    location_description: str
    estimated_repair_cost: float
    coordinates: List[Tuple[float, float]] = field(default_factory=list)
    photos: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'damage_id': self.damage_id,
            'segment_id': self.segment_id,
            'damage_type': self.damage_type,
            'severity': self.severity.value,
            'area_sq_ft': self.area_sq_ft,
            'location_description': self.location_description,
            'estimated_repair_cost': self.estimated_repair_cost,
            'coordinates': self.coordinates,
            'photos': self.photos
        }


@dataclass
class DamageAssessment:
    """Complete damage assessment"""
    assessment_id: str
    measurement_id: str
    roof_measurement: RoofMeasurement
    overall_severity: DamageSeverity
    damage_areas: List[DamageArea]
    total_damage_area_sq_ft: float
    damage_percentage: float
    estimated_repair_cost: float
    recommended_actions: List[str]
    assessment_date: datetime
    assessor_notes: List[str] = field(default_factory=list)
    confidence_score: float = 0.85
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'assessment_id': self.assessment_id,
            'measurement_id': self.measurement_id,
            'roof_measurement': self.roof_measurement.to_dict(),
            'overall_severity': self.overall_severity.value,
            'damage_areas': [area.to_dict() for area in self.damage_areas],
            'total_damage_area_sq_ft': self.total_damage_area_sq_ft,
            'damage_percentage': self.damage_percentage,
            'estimated_repair_cost': self.estimated_repair_cost,
            'recommended_actions': self.recommended_actions,
            'assessment_date': self.assessment_date.isoformat(),
            'assessor_notes': self.assessor_notes,
            'confidence_score': self.confidence_score
        }