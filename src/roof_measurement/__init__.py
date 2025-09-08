"""
Hail Hero Roof Measurement System

A comprehensive roof measurement and aerial imagery analysis system
for automated roof measurements, material estimation, and damage assessment.
"""

__version__ = "1.0.0"
__author__ = "Hail Hero Team"

from .core import RoofMeasurementSystem
from .models import RoofMeasurement, MaterialEstimate, DamageAssessment
from .imagery import AerialImageryProvider
from .detection import RoofDetector
from .measurement import RoofCalculator
from .estimation import MaterialEstimator
from .damage import DamageAssessor
from .reporting import ReportGenerator

__all__ = [
    'RoofMeasurementSystem',
    'RoofMeasurement',
    'MaterialEstimate', 
    'DamageAssessment',
    'AerialImageryProvider',
    'RoofDetector',
    'RoofCalculator',
    'MaterialEstimator',
    'DamageAssessor',
    'ReportGenerator'
]