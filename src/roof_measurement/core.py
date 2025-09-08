"""
Core roof measurement system orchestrator
"""

import uuid
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import logging

from .models import (
    RoofMeasurement, MaterialEstimate, DamageAssessment, 
    PropertyLocation, RoofSegment, RoofType, MaterialType
)
from .imagery import AerialImageryProvider
from .detection import RoofDetector
from .measurement import RoofCalculator
from .estimation import MaterialEstimator
from .damage import DamageAssessor
from .reporting import ReportGenerator
from .validation import MeasurementValidator


class RoofMeasurementSystem:
    """Main orchestrator for roof measurement and analysis"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the roof measurement system
        
        Args:
            config: Configuration dictionary with system settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.imagery_provider = AerialImageryProvider(self.config.get('imagery', {}))
        self.roof_detector = RoofDetector(self.config.get('detection', {}))
        self.roof_calculator = RoofCalculator(self.config.get('measurement', {}))
        self.material_estimator = MaterialEstimator(self.config.get('estimation', {}))
        self.damage_assessor = DamageAssessor(self.config.get('damage', {}))
        self.report_generator = ReportGenerator(self.config.get('reporting', {}))
        self.validator = MeasurementValidator(self.config.get('validation', {}))
        
        # Measurement cache
        self.measurements: Dict[str, RoofMeasurement] = {}
        
        self.logger.info("Roof Measurement System initialized")
    
    async def measure_roof(
        self, 
        address: str,
        city: str,
        state: str,
        zip_code: str,
        material_type: Optional[MaterialType] = None,
        force_refresh: bool = False
    ) -> RoofMeasurement:
        """
        Perform complete roof measurement for a property
        
        Args:
            address: Property address
            city: City
            state: State
            zip_code: ZIP code
            material_type: Optional roof material type
            force_refresh: Force re-measurement even if cached
            
        Returns:
            RoofMeasurement: Complete measurement data
        """
        measurement_id = f"rm_{uuid.uuid4().hex[:8]}"
        
        self.logger.info(f"Starting roof measurement for {address}, {city}, {state}")
        
        try:
            # Get property location
            property_location = await self._get_property_location(
                address, city, state, zip_code
            )
            
            # Check cache if not forcing refresh
            cache_key = f"{property_location.latitude}_{property_location.longitude}"
            if not force_refresh and cache_key in self.measurements:
                self.logger.info(f"Returning cached measurement for {cache_key}")
                return self.measurements[cache_key]
            
            # Get aerial imagery
            imagery_data = await self.imagery_provider.get_imagery(
                property_location.latitude,
                property_location.longitude
            )
            
            # Detect roof segments
            roof_segments = await self.roof_detector.detect_roofs(
                imagery_data,
                property_location
            )
            
            # Calculate measurements
            roof_measurement = await self.roof_calculator.calculate_measurements(
                roof_segments,
                property_location,
                imagery_data
            )
            
            # Validate measurements
            validation_result = await self.validator.validate_measurements(
                roof_measurement
            )
            
            # Apply validation corrections
            if not validation_result['is_valid']:
                roof_measurement = await self._apply_validation_corrections(
                    roof_measurement,
                    validation_result
                )
            
            # Store in cache
            self.measurements[cache_key] = roof_measurement
            
            self.logger.info(f"Completed roof measurement: {measurement_id}")
            return roof_measurement
            
        except Exception as e:
            self.logger.error(f"Error in roof measurement: {str(e)}")
            raise
    
    async def estimate_materials(
        self, 
        roof_measurement: RoofMeasurement,
        material_type: MaterialType
    ) -> MaterialEstimate:
        """
        Estimate materials and costs for roof replacement/repair
        
        Args:
            roof_measurement: Roof measurement data
            material_type: Type of roofing material
            
        Returns:
            MaterialEstimate: Complete material and cost estimate
        """
        self.logger.info(f"Estimating materials for {roof_measurement.measurement_id}")
        
        try:
            material_estimate = await self.material_estimator.estimate_materials(
                roof_measurement,
                material_type
            )
            
            self.logger.info(f"Material estimate completed: {material_estimate.estimate_id}")
            return material_estimate
            
        except Exception as e:
            self.logger.error(f"Error in material estimation: {str(e)}")
            raise
    
    async def assess_damage(
        self,
        roof_measurement: RoofMeasurement,
        hail_size: float,
        wind_speed: float,
        imagery_data: Optional[Dict[str, Any]] = None
    ) -> DamageAssessment:
        """
        Assess potential hail damage to roof
        
        Args:
            roof_measurement: Roof measurement data
            hail_size: Hail size in inches
            wind_speed: Wind speed in mph
            imagery_data: Optional post-storm imagery data
            
        Returns:
            DamageAssessment: Complete damage assessment
        """
        self.logger.info(f"Assessing damage for {roof_measurement.measurement_id}")
        
        try:
            damage_assessment = await self.damage_assessor.assess_damage(
                roof_measurement,
                hail_size,
                wind_speed,
                imagery_data
            )
            
            self.logger.info(f"Damage assessment completed: {damage_assessment.assessment_id}")
            return damage_assessment
            
        except Exception as e:
            self.logger.error(f"Error in damage assessment: {str(e)}")
            raise
    
    async def generate_measurement_report(
        self,
        roof_measurement: RoofMeasurement,
        material_estimate: Optional[MaterialEstimate] = None,
        damage_assessment: Optional[DamageAssessment] = None,
        report_format: str = "pdf"
    ) -> bytes:
        """
        Generate comprehensive measurement report
        
        Args:
            roof_measurement: Roof measurement data
            material_estimate: Optional material estimate
            damage_assessment: Optional damage assessment
            report_format: Output format ("pdf", "html", "json")
            
        Returns:
            bytes: Report data
        """
        self.logger.info(f"Generating report for {roof_measurement.measurement_id}")
        
        try:
            report_data = await self.report_generator.generate_report(
                roof_measurement,
                material_estimate,
                damage_assessment,
                report_format
            )
            
            self.logger.info(f"Report generated successfully")
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error in report generation: {str(e)}")
            raise
    
    async def _get_property_location(
        self,
        address: str,
        city: str,
        state: str,
        zip_code: str
    ) -> PropertyLocation:
        """Get property location with coordinates"""
        # This would integrate with geocoding service
        # For now, using placeholder implementation
        import random
        
        # Generate mock coordinates based on address
        lat_base = 39.7392  # Denver base
        lon_base = -104.9903
        
        lat = lat_base + (random.random() - 0.5) * 0.1
        lon = lon_base + (random.random() - 0.5) * 0.1
        
        return PropertyLocation(
            address=address,
            city=city,
            state=state,
            zip_code=zip_code,
            latitude=lat,
            longitude=lon,
            country="USA",
            elevation=random.uniform(5000, 6000)  # Denver area elevation
        )
    
    async def _apply_validation_corrections(
        self,
        measurement: RoofMeasurement,
        validation_result: Dict[str, Any]
    ) -> RoofMeasurement:
        """Apply validation corrections to measurement"""
        # This would implement various correction algorithms
        # For now, just adjust accuracy score
        measurement.measurement_accuracy *= 0.9
        measurement.validation_notes.extend(
            validation_result.get('corrections_applied', [])
        )
        return measurement
    
    def get_measurement_history(
        self,
        limit: int = 100
    ) -> List[RoofMeasurement]:
        """Get recent measurement history"""
        return sorted(
            self.measurements.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
    
    def get_measurement_by_id(
        self,
        measurement_id: str
    ) -> Optional[RoofMeasurement]:
        """Get measurement by ID"""
        for measurement in self.measurements.values():
            if measurement.measurement_id == measurement_id:
                return measurement
        return None
    
    def clear_cache(self):
        """Clear measurement cache"""
        self.measurements.clear()
        self.logger.info("Measurement cache cleared")