"""
Material estimation and cost calculation
"""

import math
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import uuid
from datetime import datetime
from enum import Enum

from .models import MaterialEstimate, MaterialItem, RoofMeasurement, MaterialType, RoofType


class WasteFactor(Enum):
    """Waste factors for different materials and roof types"""
    ASPHALT_SHINGLES = 0.10
    METAL = 0.05
    TILE = 0.15
    SLATE = 0.20
    WOOD_SHINGLES = 0.15
    RUBBER = 0.08
    BUILT_UP = 0.05
    MODIFIED_BITUMEN = 0.07


@dataclass
class MaterialPricing:
    """Material pricing information"""
    material_name: str
    unit_cost: float
    unit_type: str
    waste_factor: float
    category: str
    minimum_quantity: float = 0
    bulk_discount_threshold: float = 0
    bulk_discount_rate: float = 0


class MaterialEstimator:
    """Advanced material estimation and cost calculation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize material estimator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Pricing database (in production, this would come from an API or database)
        self.pricing_database = self._initialize_pricing_database()
        
        # Regional pricing factors
        self.regional_factors = config.get('regional_factors', {
            'northeast': 1.15,
            'southeast': 0.95,
            'midwest': 1.0,
            'southwest': 1.05,
            'west': 1.25,
            'default': 1.0
        })
        
        # Labor rates
        self.labor_rates = config.get('labor_rates', {
            'asphalt_shingles': 75.0,    # per square
            'metal': 120.0,              # per square
            'tile': 150.0,               # per square
            'slate': 200.0,              # per square
            'wood_shingles': 100.0,      # per square
            'rubber': 90.0,              # per square
            'built_up': 80.0,            # per square
            'modified_bitumen': 85.0     # per square
        })
        
        # Market conditions adjustment
        self.market_conditions = config.get('market_conditions', {
            'normal': 1.0,
            'high_demand': 1.15,
            'low_demand': 0.90,
            'supply_shortage': 1.25,
            'favorable': 0.85
        })
        
        # Overhead and profit factors
        self.overhead_factor = config.get('overhead_factor', 0.15)  # 15%
        self.profit_factor = config.get('profit_factor', 0.20)      # 20%
        
        self.logger.info("MaterialEstimator initialized")
    
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
        try:
            estimate_id = f"est_{uuid.uuid4().hex[:8]}"
            
            # Calculate regional pricing factor
            regional_factor = self._get_regional_factor(roof_measurement.property_location)
            
            # Get market conditions
            market_condition = self._get_market_conditions()
            market_factor = self.market_conditions.get(market_condition, 1.0)
            
            # Calculate materials needed
            materials = await self._calculate_materials_needed(
                roof_measurement, material_type, regional_factor, market_factor
            )
            
            # Calculate labor cost
            labor_cost = self._calculate_labor_cost(
                roof_measurement, material_type, regional_factor, market_factor
            )
            
            # Calculate total costs
            material_cost = sum(item.total_cost for item in materials)
            subtotal = material_cost + labor_cost
            
            # Apply overhead and profit
            overhead_cost = subtotal * self.overhead_factor
            profit_cost = (subtotal + overhead_cost) * self.profit_factor
            
            total_cost = subtotal + overhead_cost + profit_cost
            
            # Calculate cost per square foot
            cost_per_sq_ft = total_cost / roof_measurement.total_area_sq_ft
            
            # Create material estimate
            material_estimate = MaterialEstimate(
                estimate_id=estimate_id,
                measurement_id=roof_measurement.measurement_id,
                roof_measurement=roof_measurement,
                materials=materials,
                labor_cost=labor_cost,
                total_cost=total_cost,
                cost_per_sq_ft=cost_per_sq_ft,
                estimate_date=datetime.now(),
                material_type=material_type,
                regional_pricing_factor=regional_factor,
                market_conditions=market_condition,
                notes=self._generate_estimate_notes(
                    roof_measurement, material_type, regional_factor, market_factor
                )
            )
            
            self.logger.info(f"Material estimate completed: {estimate_id}")
            return material_estimate
            
        except Exception as e:
            self.logger.error(f"Error in material estimation: {str(e)}")
            raise
    
    async def _calculate_materials_needed(
        self,
        roof_measurement: RoofMeasurement,
        material_type: MaterialType,
        regional_factor: float,
        market_factor: float
    ) -> List[MaterialItem]:
        """Calculate materials needed for the roof"""
        
        materials = []
        total_area = roof_measurement.total_area_sq_ft
        total_perimeter = roof_measurement.total_perimeter_ft
        
        # Get material-specific calculations
        if material_type == MaterialType.ASPHALT_SHINGLES:
            materials.extend(self._calculate_asphalt_materials(total_area, total_perimeter, regional_factor, market_factor))
        elif material_type == MaterialType.METAL:
            materials.extend(self._calculate_metal_materials(total_area, total_perimeter, regional_factor, market_factor))
        elif material_type == MaterialType.TILE:
            materials.extend(self._calculate_tile_materials(total_area, total_perimeter, regional_factor, market_factor))
        elif material_type == MaterialType.SLATE:
            materials.extend(self._calculate_slate_materials(total_area, total_perimeter, regional_factor, market_factor))
        elif material_type == MaterialType.WOOD_SHINGLES:
            materials.extend(self._calculate_wood_materials(total_area, total_perimeter, regional_factor, market_factor))
        elif material_type == MaterialType.RUBBER:
            materials.extend(self._calculate_rubber_materials(total_area, total_perimeter, regional_factor, market_factor))
        elif material_type == MaterialType.BUILT_UP:
            materials.extend(self._calculate_built_up_materials(total_area, total_perimeter, regional_factor, market_factor))
        elif material_type == MaterialType.MODIFIED_BITUMEN:
            materials.extend(self._calculate_modified_bitumen_materials(total_area, total_perimeter, regional_factor, market_factor))
        
        # Add general materials needed for all roof types
        materials.extend(self._calculate_general_materials(total_area, total_perimeter, regional_factor, market_factor))
        
        return materials
    
    def _calculate_asphalt_materials(
        self,
        total_area: float,
        total_perimeter: float,
        regional_factor: float,
        market_factor: float
    ) -> List[MaterialItem]:
        """Calculate asphalt shingle materials"""
        materials = []
        
        # Roofing squares (1 square = 100 sq ft)
        squares_needed = math.ceil(total_area / 100)
        
        # Asphalt shingles
        pricing = self.pricing_database['asphalt_shingles']
        base_cost = pricing.unit_cost * regional_factor * market_factor
        
        # Apply bulk discount if applicable
        if squares_needed >= pricing.bulk_discount_threshold:
            base_cost *= (1 - pricing.bulk_discount_rate)
        
        shingles = MaterialItem(
            material_name="Asphalt Shingles (3-Tab)",
            unit_type="squares",
            quantity=squares_needed,
            unit_cost=base_cost,
            total_cost=squares_needed * base_cost,
            waste_factor=WasteFactor.ASPHALT_SHINGLES.value,
            category="roofing"
        )
        materials.append(shingles)
        
        # Underlayment
        underlayment_pricing = self.pricing_database['felt_underlayment']
        underlayment_cost = underlayment_pricing.unit_cost * regional_factor * market_factor
        
        underlayment = MaterialItem(
            material_name="15# Felt Underlayment",
            unit_type="squares",
            quantity=squares_needed,
            unit_cost=underlayment_cost,
            total_cost=squares_needed * underlayment_cost,
            waste_factor=0.05,
            category="underlayment"
        )
        materials.append(underlayment)
        
        # Drip edge
        drip_edge_pricing = self.pricing_database['drip_edge']
        drip_edge_cost = drip_edge_pricing.unit_cost * regional_factor * market_factor
        
        drip_edge = MaterialItem(
            material_name="Aluminum Drip Edge",
            unit_type="linear_ft",
            quantity=total_perimeter * 1.1,  # 10% waste
            unit_cost=drip_edge_cost,
            total_cost=total_perimeter * 1.1 * drip_edge_cost,
            waste_factor=0.10,
            category="accessories"
        )
        materials.append(drip_edge)
        
        # Roofing nails
        nails_pricing = self.pricing_database['roofing_nails']
        nails_cost = nails_pricing.unit_cost * regional_factor * market_factor
        
        nails = MaterialItem(
            material_name="Roofing Nails (1-1/4\")",
            unit_type="lbs",
            quantity=squares_needed * 2.5,  # 2.5 lbs per square
            unit_cost=nails_cost,
            total_cost=squares_needed * 2.5 * nails_cost,
            waste_factor=0.05,
            category="fasteners"
        )
        materials.append(nails)
        
        # Ridge caps
        ridge_length = total_perimeter * 0.3  # Estimate 30% of perimeter is ridge
        ridge_pricing = self.pricing_database['ridge_caps']
        ridge_cost = ridge_pricing.unit_cost * regional_factor * market_factor
        
        ridge_caps = MaterialItem(
            material_name="Asphalt Ridge Caps",
            unit_type="linear_ft",
            quantity=ridge_length * 1.1,
            unit_cost=ridge_cost,
            total_cost=ridge_length * 1.1 * ridge_cost,
            waste_factor=0.10,
            category="accessories"
        )
        materials.append(ridge_caps)
        
        return materials
    
    def _calculate_metal_materials(
        self,
        total_area: float,
        total_perimeter: float,
        regional_factor: float,
        market_factor: float
    ) -> List[MaterialItem]:
        """Calculate metal roofing materials"""
        materials = []
        
        # Metal panels
        panels_needed = math.ceil(total_area / 100)  # Per square
        
        pricing = self.pricing_database['metal_panels']
        base_cost = pricing.unit_cost * regional_factor * market_factor
        
        panels = MaterialItem(
            material_name="Metal Roofing Panels (26 Gauge)",
            unit_type="squares",
            quantity=panels_needed,
            unit_cost=base_cost,
            total_cost=panels_needed * base_cost,
            waste_factor=WasteFactor.METAL.value,
            category="roofing"
        )
        materials.append(panels)
        
        # Underlayment (synthetic for metal)
        underlayment_pricing = self.pricing_database['synthetic_underlayment']
        underlayment_cost = underlayment_pricing.unit_cost * regional_factor * market_factor
        
        underlayment = MaterialItem(
            material_name="Synthetic Underlayment",
            unit_type="squares",
            quantity=panels_needed,
            unit_cost=underlayment_cost,
            total_cost=panels_needed * underlayment_cost,
            waste_factor=0.05,
            category="underlayment"
        )
        materials.append(underlayment)
        
        # Metal screws
        screws_pricing = self.pricing_database['metal_screws']
        screws_cost = screws_pricing.unit_cost * regional_factor * market_factor
        
        screws = MaterialItem(
            material_name="Metal Roofing Screws",
            unit_type="each",
            quantity=panels_needed * 80,  # 80 screws per square
            unit_cost=screws_cost,
            total_cost=panels_needed * 80 * screws_cost,
            waste_factor=0.10,
            category="fasteners"
        )
        materials.append(screws)
        
        # Ridge vent
        ridge_length = total_perimeter * 0.3
        ridge_pricing = self.pricing_database['metal_ridge_vent']
        ridge_cost = ridge_pricing.unit_cost * regional_factor * market_factor
        
        ridge_vent = MaterialItem(
            material_name="Metal Ridge Vent",
            unit_type="linear_ft",
            quantity=ridge_length,
            unit_cost=ridge_cost,
            total_cost=ridge_length * ridge_cost,
            waste_factor=0.05,
            category="accessories"
        )
        materials.append(ridge_vent)
        
        return materials
    
    def _calculate_tile_materials(
        self,
        total_area: float,
        total_perimeter: float,
        regional_factor: float,
        market_factor: float
    ) -> List[MaterialItem]:
        """Calculate tile roofing materials"""
        materials = []
        
        # Roof tiles
        tiles_needed = math.ceil(total_area / 100)
        
        pricing = self.pricing_database['concrete_tiles']
        base_cost = pricing.unit_cost * regional_factor * market_factor
        
        tiles = MaterialItem(
            material_name="Concrete Roof Tiles",
            unit_type="squares",
            quantity=tiles_needed,
            unit_cost=base_cost,
            total_cost=tiles_needed * base_cost,
            waste_factor=WasteFactor.TILE.value,
            category="roofing"
        )
        materials.append(tiles)
        
        # Underlayment (double layer for tile)
        underlayment_pricing = self.pricing_database['synthetic_underlayment']
        underlayment_cost = underlayment_pricing.unit_cost * regional_factor * market_factor
        
        underlayment = MaterialItem(
            material_name="Synthetic Underlayment (Double Layer)",
            unit_type="squares",
            quantity=tiles_needed * 2,
            unit_cost=underlayment_cost,
            total_cost=tiles_needed * 2 * underlayment_cost,
            waste_factor=0.10,
            category="underlayment"
        )
        materials.append(underlayment)
        
        # Tile nails
        nails_pricing = self.pricing_database['tile_nails']
        nails_cost = nails_pricing.unit_cost * regional_factor * market_factor
        
        nails = MaterialItem(
            material_name="Tile Nails (2-1/2\")",
            unit_type="lbs",
            quantity=tiles_needed * 5,
            unit_cost=nails_cost,
            total_cost=tiles_needed * 5 * nails_cost,
            waste_factor=0.10,
            category="fasteners"
        )
        materials.append(nails)
        
        return materials
    
    def _calculate_slate_materials(
        self,
        total_area: float,
        total_perimeter: float,
        regional_factor: float,
        market_factor: float
    ) -> List[MaterialItem]:
        """Calculate slate roofing materials"""
        materials = []
        
        # Slate tiles
        slate_needed = math.ceil(total_area / 100)
        
        pricing = self.pricing_database['slate_tiles']
        base_cost = pricing.unit_cost * regional_factor * market_factor
        
        slate = MaterialItem(
            material_name="Slate Roof Tiles",
            unit_type="squares",
            quantity=slate_needed,
            unit_cost=base_cost,
            total_cost=slate_needed * base_cost,
            waste_factor=WasteFactor.SLATE.value,
            category="roofing"
        )
        materials.append(slate)
        
        # Copper flashing
        flashing_pricing = self.pricing_database['copper_flashing']
        flashing_cost = flashing_pricing.unit_cost * regional_factor * market_factor
        
        flashing = MaterialItem(
            material_name="Copper Flashing",
            unit_type="linear_ft",
            quantity=total_perimeter * 1.2,
            unit_cost=flashing_cost,
            total_cost=total_perimeter * 1.2 * flashing_cost,
            waste_factor=0.20,
            category="accessories"
        )
        materials.append(flashing)
        
        # Slate hooks
        hooks_pricing = self.pricing_database['slate_hooks']
        hooks_cost = hooks_pricing.unit_cost * regional_factor * market_factor
        
        hooks = MaterialItem(
            material_name="Slate Hooks",
            unit_type="each",
            quantity=slate_needed * 300,  # 300 hooks per square
            unit_cost=hooks_cost,
            total_cost=slate_needed * 300 * hooks_cost,
            waste_factor=0.15,
            category="fasteners"
        )
        materials.append(hooks)
        
        return materials
    
    def _calculate_wood_materials(
        self,
        total_area: float,
        total_perimeter: float,
        regional_factor: float,
        market_factor: float
    ) -> List[MaterialItem]:
        """Calculate wood shingle materials"""
        materials = []
        
        # Wood shingles
        shingles_needed = math.ceil(total_area / 100)
        
        pricing = self.pricing_database['wood_shingles']
        base_cost = pricing.unit_cost * regional_factor * market_factor
        
        shingles = MaterialItem(
            material_name="Cedar Wood Shingles",
            unit_type="squares",
            quantity=shingles_needed,
            unit_cost=base_cost,
            total_cost=shingles_needed * base_cost,
            waste_factor=WasteFactor.WOOD_SHINGLES.value,
            category="roofing"
        )
        materials.append(shingles)
        
        # Underlayment (breathable for wood)
        underlayment_pricing = self.pricing_database['breathable_underlayment']
        underlayment_cost = underlayment_pricing.unit_cost * regional_factor * market_factor
        
        underlayment = MaterialItem(
            material_name="Breathable Underlayment",
            unit_type="squares",
            quantity=shingles_needed,
            unit_cost=underlayment_cost,
            total_cost=shingles_needed * underlayment_cost,
            waste_factor=0.05,
            category="underlayment"
        )
        materials.append(underlayment)
        
        # Stainless steel staples
        staples_pricing = self.pricing_database['stainless_staples']
        staples_cost = staples_pricing.unit_cost * regional_factor * market_factor
        
        staples = MaterialItem(
            material_name="Stainless Steel Staples",
            unit_type="lbs",
            quantity=shingles_needed * 2,
            unit_cost=staples_cost,
            total_cost=shingles_needed * 2 * staples_cost,
            waste_factor=0.05,
            category="fasteners"
        )
        materials.append(staples)
        
        return materials
    
    def _calculate_rubber_materials(
        self,
        total_area: float,
        total_perimeter: float,
        regional_factor: float,
        market_factor: float
    ) -> List[MaterialItem]:
        """Calculate rubber roofing materials"""
        materials = []
        
        # EPDM rubber membrane
        rubber_needed = math.ceil(total_area / 100)
        
        pricing = self.pricing_database['epdm_membrane']
        base_cost = pricing.unit_cost * regional_factor * market_factor
        
        rubber = MaterialItem(
            material_name="EPDM Rubber Membrane (60 mil)",
            unit_type="squares",
            quantity=rubber_needed,
            unit_cost=base_cost,
            total_cost=rubber_needed * base_cost,
            waste_factor=WasteFactor.RUBBER.value,
            category="roofing"
        )
        materials.append(rubber)
        
        # Rubber adhesive
        adhesive_pricing = self.pricing_database['rubber_adhesive']
        adhesive_cost = adhesive_pricing.unit_cost * regional_factor * market_factor
        
        adhesive = MaterialItem(
            material_name="EPDM Bonding Adhesive",
            unit_type="gallons",
            quantity=rubber_needed * 2,  # 2 gallons per square
            unit_cost=adhesive_cost,
            total_cost=rubber_needed * 2 * adhesive_cost,
            waste_factor=0.10,
            category="adhesives"
        )
        materials.append(adhesive)
        
        # Rubber flashing
        flashing_pricing = self.pricing_database['rubber_flashing']
        flashing_cost = flashing_pricing.unit_cost * regional_factor * market_factor
        
        flashing = MaterialItem(
            material_name="EPDM Flashing",
            unit_type="linear_ft",
            quantity=total_perimeter * 1.1,
            unit_cost=flashing_cost,
            total_cost=total_perimeter * 1.1 * flashing_cost,
            waste_factor=0.10,
            category="accessories"
        )
        materials.append(flashing)
        
        return materials
    
    def _calculate_built_up_materials(
        self,
        total_area: float,
        total_perimeter: float,
        regional_factor: float,
        market_factor: float
    ) -> List[MaterialItem]:
        """Calculate built-up roofing materials"""
        materials = []
        
        # Built-up roofing (3-ply)
        bur_needed = math.ceil(total_area / 100)
        
        pricing = self.pricing_database['bur_membrane']
        base_cost = pricing.unit_cost * regional_factor * market_factor
        
        bur = MaterialItem(
            material_name="Built-Up Roofing (3-Ply)",
            unit_type="squares",
            quantity=bur_needed,
            unit_cost=base_cost,
            total_cost=bur_needed * base_cost,
            waste_factor=WasteFactor.BUILT_UP.value,
            category="roofing"
        )
        materials.append(bur)
        
        # Asphalt
        asphalt_pricing = self.pricing_database['roofing_asphalt']
        asphalt_cost = asphalt_pricing.unit_cost * regional_factor * market_factor
        
        asphalt = MaterialItem(
            material_name="Roofing Asphalt (Type III)",
            unit_type="gallons",
            quantity=bur_needed * 8,  # 8 gallons per square
            unit_cost=asphalt_cost,
            total_cost=bur_needed * 8 * asphalt_cost,
            waste_factor=0.05,
            category="adhesives"
        )
        materials.append(asphalt)
        
        # Gravel surfacing
        gravel_pricing = self.pricing_database['roofing_gravel']
        gravel_cost = gravel_pricing.unit_cost * regional_factor * market_factor
        
        gravel = MaterialItem(
            material_name="Roofing Gravel",
            unit_type="tons",
            quantity=bur_needed * 0.5,  # 0.5 tons per square
            unit_cost=gravel_cost,
            total_cost=bur_needed * 0.5 * gravel_cost,
            waste_factor=0.05,
            category="surfacing"
        )
        materials.append(gravel)
        
        return materials
    
    def _calculate_modified_bitumen_materials(
        self,
        total_area: float,
        total_perimeter: float,
        regional_factor: float,
        market_factor: float
    ) -> List[MaterialItem]:
        """Calculate modified bitumen materials"""
        materials = []
        
        # Modified bitumen membrane
        mb_needed = math.ceil(total_area / 100)
        
        pricing = self.pricing_database['modified_bitumen']
        base_cost = pricing.unit_cost * regional_factor * market_factor
        
        mb = MaterialItem(
            material_name="Modified Bitumen Membrane (APP)",
            unit_type="squares",
            quantity=mb_needed,
            unit_cost=base_cost,
            total_cost=mb_needed * base_cost,
            waste_factor=WasteFactor.MODIFIED_BITUMEN.value,
            category="roofing"
        )
        materials.append(mb)
        
        # Base sheet
        base_pricing = self.pricing_database['base_sheet']
        base_cost = base_pricing.unit_cost * regional_factor * market_factor
        
        base_sheet = MaterialItem(
            material_name="Base Sheet",
            unit_type="squares",
            quantity=mb_needed,
            unit_cost=base_cost,
            total_cost=mb_needed * base_cost,
            waste_factor=0.05,
            category="underlayment"
        )
        materials.append(base_sheet)
        
        # Adhesive
        adhesive_pricing = self.pricing_database['mb_adhesive']
        adhesive_cost = adhesive_pricing.unit_cost * regional_factor * market_factor
        
        adhesive = MaterialItem(
            material_name="Modified Bitumen Adhesive",
            unit_type="gallons",
            quantity=mb_needed * 1.5,
            unit_cost=adhesive_cost,
            total_cost=mb_needed * 1.5 * adhesive_cost,
            waste_factor=0.10,
            category="adhesives"
        )
        materials.append(adhesive)
        
        return materials
    
    def _calculate_general_materials(
        self,
        total_area: float,
        total_perimeter: float,
        regional_factor: float,
        market_factor: float
    ) -> List[MaterialItem]:
        """Calculate general materials needed for all roof types"""
        materials = []
        
        # Roof cement
        cement_pricing = self.pricing_database['roofing_cement']
        cement_cost = cement_pricing.unit_cost * regional_factor * market_factor
        
        cement = MaterialItem(
            material_name="Roofing Cement",
            unit_type="gallons",
            quantity=math.ceil(total_area / 1000),  # 1 gallon per 1000 sq ft
            unit_cost=cement_cost,
            total_cost=math.ceil(total_area / 1000) * cement_cost,
            waste_factor=0.10,
            category="sealants"
        )
        materials.append(cement)
        
        # Caulk
        caulk_pricing = self.pricing_database['roof_caulk']
        caulk_cost = caulk_pricing.unit_cost * regional_factor * market_factor
        
        caulk = MaterialItem(
            material_name="Roofing Caulk",
            unit_type="tubes",
            quantity=math.ceil(total_perimeter / 50),  # 1 tube per 50 linear ft
            unit_cost=caulk_cost,
            total_cost=math.ceil(total_perimeter / 50) * caulk_cost,
            waste_factor=0.05,
            category="sealants"
        )
        materials.append(caulk)
        
        # Safety equipment (per job)
        safety_pricing = self.pricing_database['safety_equipment']
        safety_cost = safety_pricing.unit_cost * regional_factor * market_factor
        
        safety = MaterialItem(
            material_name="Safety Equipment Rental",
            unit_type="each",
            quantity=1,
            unit_cost=safety_cost,
            total_cost=safety_cost,
            waste_factor=0.0,
            category="equipment"
        )
        materials.append(safety)
        
        # Cleanup and disposal
        cleanup_pricing = self.pricing_database['cleanup_disposal']
        cleanup_cost = cleanup_pricing.unit_cost * regional_factor * market_factor
        
        cleanup = MaterialItem(
            material_name="Cleanup and Disposal",
            unit_type="each",
            quantity=1,
            unit_cost=cleanup_cost,
            total_cost=cleanup_cost,
            waste_factor=0.0,
            category="services"
        )
        materials.append(cleanup)
        
        return materials
    
    def _calculate_labor_cost(
        self,
        roof_measurement: RoofMeasurement,
        material_type: MaterialType,
        regional_factor: float,
        market_factor: float
    ) -> float:
        """Calculate labor cost based on roof complexity and material type"""
        
        # Base labor rate per square
        base_rate = self.labor_rates.get(material_type.value, 100.0)
        
        # Adjust for regional factors
        regional_rate = base_rate * regional_factor
        
        # Adjust for market conditions
        market_rate = regional_rate * market_factor
        
        # Calculate number of squares
        squares = roof_measurement.total_area_sq_ft / 100
        
        # Complexity factor
        complexity_factor = roof_measurement.roof_complexity
        
        # Pitch factor (steeper roofs cost more)
        pitch_factor = 1.0 + (roof_measurement.average_pitch_degrees / 100)
        
        # Accessibility factor (based on height)
        height_factor = 1.0 + (roof_measurement.estimated_height_ft / 100)
        
        # Total labor cost
        labor_cost = squares * market_rate * complexity_factor * pitch_factor * height_factor
        
        return labor_cost
    
    def _get_regional_factor(self, property_location) -> float:
        """Get regional pricing factor based on property location"""
        # Simple regional mapping based on state
        state = property_location.state.upper()
        
        northeast_states = {'ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'}
        southeast_states = {'MD', 'DE', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'LA', 'AR'}
        midwest_states = {'OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'}
        southwest_states = {'TX', 'OK', 'NM', 'AZ'}
        west_states = {'CO', 'WY', 'MT', 'ID', 'UT', 'NV', 'CA', 'OR', 'WA', 'AK', 'HI'}
        
        if state in northeast_states:
            return self.regional_factors['northeast']
        elif state in southeast_states:
            return self.regional_factors['southeast']
        elif state in midwest_states:
            return self.regional_factors['midwest']
        elif state in southwest_states:
            return self.regional_factors['southwest']
        elif state in west_states:
            return self.regional_factors['west']
        else:
            return self.regional_factors['default']
    
    def _get_market_conditions(self) -> str:
        """Get current market conditions"""
        # In production, this would come from an economic API
        # For now, return normal conditions
        return "normal"
    
    def _generate_estimate_notes(
        self,
        roof_measurement: RoofMeasurement,
        material_type: MaterialType,
        regional_factor: float,
        market_factor: float
    ) -> List[str]:
        """Generate notes for the estimate"""
        notes = []
        
        # General notes
        notes.append(f"Roof area: {roof_measurement.total_area_sq_ft:.0f} sq ft")
        notes.append(f"Roof pitch: {roof_measurement.average_pitch_degrees:.1f}°")
        notes.append(f"Complexity factor: {roof_measurement.roof_complexity:.2f}")
        
        # Material-specific notes
        if material_type == MaterialType.ASPHALT_SHINGLES:
            notes.append("Standard 3-tab asphalt shingles with 15# felt underlayment")
        elif material_type == MaterialType.METAL:
            notes.append("26-gauge standing seam metal roofing")
        elif material_type == MaterialType.TILE:
            notes.append("Concrete tile roofing with double underlayment")
        elif material_type == MaterialType.SLATE:
            notes.append("Natural slate roofing with copper flashing")
        
        # Pricing notes
        if regional_factor != 1.0:
            notes.append(f"Regional pricing adjustment: {regional_factor:.2f}x")
        
        if market_factor != 1.0:
            notes.append(f"Market conditions adjustment: {market_factor:.2f}x")
        
        # Warranty notes
        notes.append("Material warranty varies by manufacturer")
        notes.append("Workmanship warranty: 2-10 years depending on contractor")
        
        return notes
    
    def _initialize_pricing_database(self) -> Dict[str, MaterialPricing]:
        """Initialize material pricing database"""
        return {
            # Asphalt shingles
            'asphalt_shingles': MaterialPricing(
                material_name="Asphalt Shingles (3-Tab)",
                unit_cost=85.0,
                unit_type="square",
                waste_factor=0.10,
                category="roofing",
                bulk_discount_threshold=50,
                bulk_discount_rate=0.05
            ),
            'felt_underlayment': MaterialPricing(
                material_name="15# Felt Underlayment",
                unit_cost=25.0,
                unit_type="square",
                waste_factor=0.05,
                category="underlayment"
            ),
            'drip_edge': MaterialPricing(
                material_name="Aluminum Drip Edge",
                unit_cost=2.50,
                unit_type="linear_ft",
                waste_factor=0.10,
                category="accessories"
            ),
            'roofing_nails': MaterialPricing(
                material_name="Roofing Nails (1-1/4\")",
                unit_cost=5.00,
                unit_type="lbs",
                waste_factor=0.05,
                category="fasteners"
            ),
            'ridge_caps': MaterialPricing(
                material_name="Asphalt Ridge Caps",
                unit_cost=15.00,
                unit_type="linear_ft",
                waste_factor=0.10,
                category="accessories"
            ),
            
            # Metal roofing
            'metal_panels': MaterialPricing(
                material_name="Metal Roofing Panels (26 Gauge)",
                unit_cost=350.0,
                unit_type="square",
                waste_factor=0.05,
                category="roofing",
                bulk_discount_threshold=30,
                bulk_discount_rate=0.03
            ),
            'synthetic_underlayment': MaterialPricing(
                material_name="Synthetic Underlayment",
                unit_cost=85.0,
                unit_type="square",
                waste_factor=0.05,
                category="underlayment"
            ),
            'metal_screws': MaterialPricing(
                material_name="Metal Roofing Screws",
                unit_cost=0.15,
                unit_type="each",
                waste_factor=0.10,
                category="fasteners"
            ),
            'metal_ridge_vent': MaterialPricing(
                material_name="Metal Ridge Vent",
                unit_cost=25.00,
                unit_type="linear_ft",
                waste_factor=0.05,
                category="accessories"
            ),
            
            # Tile roofing
            'concrete_tiles': MaterialPricing(
                material_name="Concrete Roof Tiles",
                unit_cost=450.0,
                unit_type="square",
                waste_factor=0.15,
                category="roofing"
            ),
            'tile_nails': MaterialPricing(
                material_name="Tile Nails (2-1/2\")",
                unit_cost=12.00,
                unit_type="lbs",
                waste_factor=0.10,
                category="fasteners"
            ),
            
            # Slate roofing
            'slate_tiles': MaterialPricing(
                material_name="Slate Roof Tiles",
                unit_cost=1200.0,
                unit_type="square",
                waste_factor=0.20,
                category="roofing"
            ),
            'copper_flashing': MaterialPricing(
                material_name="Copper Flashing",
                unit_cost=45.00,
                unit_type="linear_ft",
                waste_factor=0.20,
                category="accessories"
            ),
            'slate_hooks': MaterialPricing(
                material_name="Slate Hooks",
                unit_cost=0.35,
                unit_type="each",
                waste_factor=0.15,
                category="fasteners"
            ),
            
            # Wood shingles
            'wood_shingles': MaterialPricing(
                material_name="Cedar Wood Shingles",
                unit_cost=250.0,
                unit_type="square",
                waste_factor=0.15,
                category="roofing"
            ),
            'breathable_underlayment': MaterialPricing(
                material_name="Breathable Underlayment",
                unit_cost=110.0,
                unit_type="square",
                waste_factor=0.05,
                category="underlayment"
            ),
            'stainless_staples': MaterialPricing(
                material_name="Stainless Steel Staples",
                unit_cost=8.00,
                unit_type="lbs",
                waste_factor=0.05,
                category="fasteners"
            ),
            
            # Rubber roofing
            'epdm_membrane': MaterialPricing(
                material_name="EPDM Rubber Membrane (60 mil)",
                unit_cost=280.0,
                unit_type="square",
                waste_factor=0.08,
                category="roofing"
            ),
            'rubber_adhesive': MaterialPricing(
                material_name="EPDM Bonding Adhesive",
                unit_cost=85.00,
                unit_type="gallons",
                waste_factor=0.10,
                category="adhesives"
            ),
            'rubber_flashing': MaterialPricing(
                material_name="EPDM Flashing",
                unit_cost=18.00,
                unit_type="linear_ft",
                waste_factor=0.10,
                category="accessories"
            ),
            
            # Built-up roofing
            'bur_membrane': MaterialPricing(
                material_name="Built-Up Roofing (3-Ply)",
                unit_cost=180.0,
                unit_type="square",
                waste_factor=0.05,
                category="roofing"
            ),
            'roofing_asphalt': MaterialPricing(
                material_name="Roofing Asphalt (Type III)",
                unit_cost=25.00,
                unit_type="gallons",
                waste_factor=0.05,
                category="adhesives"
            ),
            'roofing_gravel': MaterialPricing(
                material_name="Roofing Gravel",
                unit_cost=150.00,
                unit_type="tons",
                waste_factor=0.05,
                category="surfacing"
            ),
            
            # Modified bitumen
            'modified_bitumen': MaterialPricing(
                material_name="Modified Bitumen Membrane (APP)",
                unit_cost=220.0,
                unit_type="square",
                waste_factor=0.07,
                category="roofing"
            ),
            'base_sheet': MaterialPricing(
                material_name="Base Sheet",
                unit_cost=65.0,
                unit_type="square",
                waste_factor=0.05,
                category="underlayment"
            ),
            'mb_adhesive': MaterialPricing(
                material_name="Modified Bitumen Adhesive",
                unit_cost=95.00,
                unit_type="gallons",
                waste_factor=0.10,
                category="adhesives"
            ),
            
            # General materials
            'roofing_cement': MaterialPricing(
                material_name="Roofing Cement",
                unit_cost=35.00,
                unit_type="gallons",
                waste_factor=0.10,
                category="sealants"
            ),
            'roof_caulk': MaterialPricing(
                material_name="Roofing Caulk",
                unit_cost=8.50,
                unit_type="tubes",
                waste_factor=0.05,
                category="sealants"
            ),
            'safety_equipment': MaterialPricing(
                material_name="Safety Equipment Rental",
                unit_cost=500.00,
                unit_type="each",
                waste_factor=0.0,
                category="equipment"
            ),
            'cleanup_disposal': MaterialPricing(
                material_name="Cleanup and Disposal",
                unit_cost=750.00,
                unit_type="each",
                waste_factor=0.0,
                category="services"
            )
        }
    
    def update_pricing_database(self, updates: Dict[str, Dict[str, Any]]):
        """Update pricing database with new values"""
        for material_name, update_data in updates.items():
            if material_name in self.pricing_database:
                pricing = self.pricing_database[material_name]
                for key, value in update_data.items():
                    if hasattr(pricing, key):
                        setattr(pricing, key, value)
    
    def get_pricing_database(self) -> Dict[str, MaterialPricing]:
        """Get current pricing database"""
        return self.pricing_database.copy()
    
    def calculate_material_costs_only(
        self,
        roof_measurement: RoofMeasurement,
        material_type: MaterialType
    ) -> Dict[str, float]:
        """Calculate only material costs (no labor or overhead)"""
        regional_factor = self._get_regional_factor(roof_measurement.property_location)
        market_factor = self.market_conditions.get(self._get_market_conditions(), 1.0)
        
        materials = self._calculate_materials_needed(
            roof_measurement, material_type, regional_factor, market_factor
        )
        
        total_material_cost = sum(item.total_cost for item in materials)
        
        return {
            'total_material_cost': total_material_cost,
            'material_cost_per_sq_ft': total_material_cost / roof_measurement.total_area_sq_ft,
            'material_breakdown': {item.material_name: item.total_cost for item in materials}
        }
    
    def get_estimation_stats(self) -> Dict[str, Any]:
        """Get estimation statistics"""
        return {
            'regional_factors': self.regional_factors,
            'labor_rates': self.labor_rates,
            'market_conditions': self.market_conditions,
            'overhead_factor': self.overhead_factor,
            'profit_factor': self.profit_factor,
            'materials_count': len(self.pricing_database)
        }