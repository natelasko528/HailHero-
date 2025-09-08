"""
Roof measurement calculations and analysis
"""

import cv2
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import uuid
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from .models import RoofMeasurement, RoofSegment, RoofType, PropertyLocation
from .detection import DetectedRoof


@dataclass
class MeasurementResult:
    """Result of roof measurement calculation"""
    segment_id: str
    area_sq_ft: float
    perimeter_ft: float
    pitch_degrees: float
    pitch_ratio: float
    complexity_factor: float
    orientation: str
    coordinates: List[Tuple[float, float]]
    confidence_score: float
    measurement_method: str
    validation_metrics: Dict[str, Any]


class RoofCalculator:
    """Advanced roof measurement calculator"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize roof calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Measurement parameters
        self.pixel_to_foot_ratio = config.get('pixel_to_foot_ratio', 0.5)  # Default 0.5 ft per pixel
        self.min_area_threshold = config.get('min_area_threshold', 50)  # Minimum area in sq ft
        self.complexity_thresholds = config.get('complexity_thresholds', {
            'simple': 1.0,
            'moderate': 1.2,
            'complex': 1.5,
            'very_complex': 2.0
        })
        
        # Pitch calculation parameters
        self.shadow_analysis_enabled = config.get('shadow_analysis_enabled', True)
        self.stereo_analysis_enabled = config.get('stereo_analysis_enabled', False)
        self.default_pitch = config.get('default_pitch', 30.0)  # degrees
        
        # Validation parameters
        self.accuracy_tolerance = config.get('accuracy_tolerance', 0.1)  # 10% tolerance
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        self.logger.info("RoofCalculator initialized")
    
    async def calculate_measurements(
        self,
        detected_roofs: List[DetectedRoof],
        property_location: PropertyLocation,
        imagery_data: Dict[str, Any]
    ) -> RoofMeasurement:
        """
        Calculate comprehensive roof measurements
        
        Args:
            detected_roofs: List of detected roof segments
            property_location: Property location information
            imagery_data: Imagery data and metadata
            
        Returns:
            RoofMeasurement: Complete measurement data
        """
        try:
            measurement_id = f"rm_{uuid.uuid4().hex[:8]}"
            
            # Calculate pixel to foot ratio based on imagery metadata
            pixel_ratio = self._calculate_pixel_ratio(imagery_data)
            
            # Process each roof segment
            segments = []
            total_area = 0
            total_perimeter = 0
            pitch_values = []
            
            for detected_roof in detected_roofs:
                # Calculate segment measurements
                measurement = await self._calculate_segment_measurements(
                    detected_roof, pixel_ratio, imagery_data
                )
                
                if measurement.area_sq_ft > self.min_area_threshold:
                    # Create roof segment
                    segment = RoofSegment(
                        segment_id=measurement.segment_id,
                        roof_type=RoofType(detected_roof.roof_type),
                        area_sq_ft=measurement.area_sq_ft,
                        perimeter_ft=measurement.perimeter_ft,
                        pitch_degrees=measurement.pitch_degrees,
                        pitch_ratio=measurement.pitch_ratio,
                        complexity_factor=measurement.complexity_factor,
                        orientation=measurement.orientation,
                        coordinates=measurement.coordinates,
                        features=measurement.validation_metrics
                    )
                    
                    segments.append(segment)
                    total_area += measurement.area_sq_ft
                    total_perimeter += measurement.perimeter_ft
                    pitch_values.append(measurement.pitch_degrees)
            
            # Calculate overall measurements
            avg_pitch = np.mean(pitch_values) if pitch_values else self.default_pitch
            avg_pitch_ratio = math.tan(math.radians(avg_pitch))
            
            # Calculate roof complexity
            roof_complexity = self._calculate_roof_complexity(segments)
            
            # Estimate building height and stories
            estimated_height = self._estimate_building_height(segments, avg_pitch)
            number_of_stories = self._estimate_number_of_stories(estimated_height)
            
            # Calculate accuracy score
            accuracy_score = self._calculate_accuracy_score(segments, imagery_data)
            
            # Create roof measurement
            roof_measurement = RoofMeasurement(
                measurement_id=measurement_id,
                property_location=property_location,
                timestamp=imagery_data.get('timestamp', imagery_data.get('metadata', {}).get('acquisition_date')),
                total_area_sq_ft=total_area,
                total_perimeter_ft=total_perimeter,
                segments=segments,
                average_pitch_degrees=avg_pitch,
                average_pitch_ratio=avg_pitch_ratio,
                roof_complexity=roof_complexity,
                estimated_height_ft=estimated_height,
                number_of_stories=number_of_stories,
                imagery_metadata=imagery_data.get('metadata', {}),
                measurement_accuracy=accuracy_score,
                quality_score=min(accuracy_score * 1.1, 1.0),  # Quality slightly higher than accuracy
                source_imagery_url=imagery_data.get('metadata', {}).get('source_imagery_url')
            )
            
            self.logger.info(f"Completed roof measurement: {measurement_id}")
            return roof_measurement
            
        except Exception as e:
            self.logger.error(f"Error in roof measurement calculation: {str(e)}")
            raise
    
    async def _calculate_segment_measurements(
        self,
        detected_roof: DetectedRoof,
        pixel_ratio: float,
        imagery_data: Dict[str, Any]
    ) -> MeasurementResult:
        """Calculate measurements for individual roof segment"""
        
        # Extract image data
        image = self._extract_image_from_imagery(imagery_data)
        
        # Calculate basic measurements
        area_sq_ft = detected_roof.area_pixels * (pixel_ratio ** 2)
        
        # Calculate perimeter
        perimeter_pixels = self._calculate_perimeter(detected_roof.polygon_points)
        perimeter_ft = perimeter_pixels * pixel_ratio
        
        # Calculate pitch using multiple methods
        pitch_result = await self._calculate_pitch(
            detected_roof, image, pixel_ratio, imagery_data
        )
        
        # Calculate complexity factor
        complexity_factor = self._calculate_segment_complexity(detected_roof)
        
        # Determine orientation
        orientation = self._determine_orientation(detected_roof.polygon_points)
        
        # Convert coordinates to real-world
        coordinates = self._convert_coordinates(
            detected_roof.polygon_points, pixel_ratio, imagery_data
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_segment_confidence(
            detected_roof, pitch_result, area_sq_ft
        )
        
        # Validation metrics
        validation_metrics = {
            'detection_confidence': detected_roof.confidence_score,
            'pitch_confidence': pitch_result.get('confidence', 0.5),
            'area_confidence': min(area_sq_ft / 100, 1.0),  # Higher confidence for larger areas
            'measurement_methods': pitch_result.get('methods_used', []),
            'pixel_ratio': pixel_ratio,
            'image_quality': self._assess_image_quality(image)
        }
        
        return MeasurementResult(
            segment_id=detected_roof.segment_id,
            area_sq_ft=area_sq_ft,
            perimeter_ft=perimeter_ft,
            pitch_degrees=pitch_result['pitch_degrees'],
            pitch_ratio=pitch_result['pitch_ratio'],
            complexity_factor=complexity_factor,
            orientation=orientation,
            coordinates=coordinates,
            confidence_score=confidence_score,
            measurement_method=pitch_result.get('primary_method', 'default'),
            validation_metrics=validation_metrics
        )
    
    def _calculate_pixel_ratio(self, imagery_data: Dict[str, Any]) -> float:
        """Calculate pixel to foot ratio from imagery metadata"""
        metadata = imagery_data.get('metadata', {})
        
        # Try to get resolution from metadata
        resolution = metadata.get('resolution') or metadata.get('gsd')
        if resolution:
            # Convert resolution (usually in meters) to feet per pixel
            resolution_ft = resolution * 3.28084
            return resolution_ft
        
        # Estimate based on zoom level and typical satellite imagery
        zoom_level = metadata.get('zoom_level', 20)
        if zoom_level >= 19:
            return 0.25  # High resolution
        elif zoom_level >= 17:
            return 0.5   # Medium resolution
        else:
            return 1.0   # Low resolution
    
    def _extract_image_from_imagery(self, imagery_data: Dict[str, Any]) -> np.ndarray:
        """Extract image array from imagery data"""
        import io
        from PIL import Image
        
        image_data = imagery_data['image_data']
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    
    def _calculate_perimeter(self, polygon_points: List[Tuple[float, float]]) -> float:
        """Calculate perimeter of polygon"""
        if len(polygon_points) < 3:
            return 0
        
        perimeter = 0
        for i in range(len(polygon_points)):
            p1 = polygon_points[i]
            p2 = polygon_points[(i + 1) % len(polygon_points)]
            distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            perimeter += distance
        
        return perimeter
    
    async def _calculate_pitch(
        self,
        detected_roof: DetectedRoof,
        image: np.ndarray,
        pixel_ratio: float,
        imagery_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate roof pitch using multiple methods"""
        
        methods = []
        pitch_values = []
        confidences = []
        
        # Method 1: Shadow analysis
        if self.shadow_analysis_enabled:
            shadow_pitch = self._calculate_pitch_from_shadows(detected_roof, image)
            if shadow_pitch['confidence'] > 0.3:
                methods.append('shadow_analysis')
                pitch_values.append(shadow_pitch['pitch'])
                confidences.append(shadow_pitch['confidence'])
        
        # Method 2: Edge analysis
        edge_pitch = self._calculate_pitch_from_edges(detected_roof, image)
        if edge_pitch['confidence'] > 0.3:
            methods.append('edge_analysis')
            pitch_values.append(edge_pitch['pitch'])
            confidences.append(edge_pitch['confidence'])
        
        # Method 3: Texture analysis
        texture_pitch = self._calculate_pitch_from_texture(detected_roof, image)
        if texture_pitch['confidence'] > 0.3:
            methods.append('texture_analysis')
            pitch_values.append(texture_pitch['pitch'])
            confidences.append(texture_pitch['confidence'])
        
        # Method 4: Roof type defaults
        if not pitch_values:
            default_pitch = self._get_default_pitch_for_type(detected_roof.roof_type)
            methods.append('default')
            pitch_values.append(default_pitch)
            confidences.append(0.5)
        
        # Weighted average of pitch values
        if pitch_values:
            total_confidence = sum(confidences)
            if total_confidence > 0:
                weighted_pitch = sum(p * c for p, c in zip(pitch_values, confidences)) / total_confidence
                overall_confidence = total_confidence / len(confidences)
            else:
                weighted_pitch = self.default_pitch
                overall_confidence = 0.5
        else:
            weighted_pitch = self.default_pitch
            overall_confidence = 0.5
        
        return {
            'pitch_degrees': weighted_pitch,
            'pitch_ratio': math.tan(math.radians(weighted_pitch)),
            'confidence': overall_confidence,
            'methods_used': methods,
            'primary_method': methods[-1] if methods else 'default',
            'individual_measurements': list(zip(methods, pitch_values, confidences))
        }
    
    def _calculate_pitch_from_shadows(
        self,
        detected_roof: DetectedRoof,
        image: np.ndarray
    ) -> Dict[str, float]:
        """Calculate pitch from shadow analysis"""
        try:
            # Extract region of interest
            x, y, w, h = detected_roof.bounding_box
            roi = image[y:y+h, x:x+w]
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            
            # Apply thresholding to find shadows
            _, shadow_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Find shadow contours
            contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Analyze shadow length and direction
                shadow_lengths = []
                for contour in contours:
                    if cv2.contourArea(contour) > 100:
                        # Get bounding rectangle
                        rect = cv2.minAreaRect(contour)
                        (cx, cy), (w_rect, h_rect), angle = rect
                        
                        # Estimate shadow length
                        shadow_length = max(w_rect, h_rect)
                        shadow_lengths.append(shadow_length)
                
                if shadow_lengths:
                    avg_shadow_length = np.mean(shadow_lengths)
                    # Estimate pitch based on shadow length and typical sun angles
                    # This is a simplified calculation
                    estimated_pitch = min(max(avg_shadow_length * 0.5, 10), 60)
                    confidence = min(len(shadow_lengths) * 0.1, 0.8)
                    
                    return {'pitch': estimated_pitch, 'confidence': confidence}
            
            return {'pitch': self.default_pitch, 'confidence': 0.2}
            
        except Exception as e:
            self.logger.warning(f"Error in shadow pitch calculation: {str(e)}")
            return {'pitch': self.default_pitch, 'confidence': 0.1}
    
    def _calculate_pitch_from_edges(
        self,
        detected_roof: DetectedRoof,
        image: np.ndarray
    ) -> Dict[str, float]:
        """Calculate pitch from edge analysis"""
        try:
            # Extract region of interest
            x, y, w, h = detected_roof.bounding_box
            roi = image[y:y+h, x:x+w]
            
            # Edge detection
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Hough line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    angles.append(angle)
                
                if angles:
                    # Analyze angle distribution
                    angles = np.array(angles)
                    angle_std = np.std(angles)
                    
                    # Estimate pitch based on angle variation
                    # More variation suggests steeper pitch
                    estimated_pitch = min(max(angle_std * 2, 15), 55)
                    confidence = min(len(lines) * 0.05, 0.7)
                    
                    return {'pitch': estimated_pitch, 'confidence': confidence}
            
            return {'pitch': self.default_pitch, 'confidence': 0.3}
            
        except Exception as e:
            self.logger.warning(f"Error in edge pitch calculation: {str(e)}")
            return {'pitch': self.default_pitch, 'confidence': 0.1}
    
    def _calculate_pitch_from_texture(
        self,
        detected_roof: DetectedRoof,
        image: np.ndarray
    ) -> Dict[str, float]:
        """Calculate pitch from texture analysis"""
        try:
            # Extract region of interest
            x, y, w, h = detected_roof.bounding_box
            roi = image[y:y+h, x:x+w]
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
            
            # Calculate texture features
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            
            # Local Binary Pattern for texture
            lbp = self._calculate_lbp(gray)
            
            # Calculate texture uniformity
            texture_variance = np.var(lbp)
            
            # Calculate color variance
            color_variance = np.var(hsv[:, :, 1])  # Saturation variance
            
            # Estimate pitch based on texture and color variation
            # Steeper roofs show more texture variation
            combined_variance = texture_variance + color_variance
            estimated_pitch = min(max(combined_variance * 0.01, 5), 45)
            confidence = min(combined_variance * 0.001, 0.6)
            
            return {'pitch': estimated_pitch, 'confidence': confidence}
            
        except Exception as e:
            self.logger.warning(f"Error in texture pitch calculation: {str(e)}")
            return {'pitch': self.default_pitch, 'confidence': 0.1}
    
    def _calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern"""
        height, width = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center = image[i, j]
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                binary = sum([1 if n >= center else 0 for n in neighbors])
                lbp[i, j] = binary
        
        return lbp
    
    def _get_default_pitch_for_type(self, roof_type: str) -> float:
        """Get default pitch based on roof type"""
        defaults = {
            'flat': 2.0,
            'hip': 25.0,
            'gable': 30.0,
            'mansard': 45.0,
            'gambrel': 35.0,
            'shed': 15.0,
            'butterfly': 20.0,
            'complex': 28.0
        }
        return defaults.get(roof_type, self.default_pitch)
    
    def _calculate_segment_complexity(self, detected_roof: DetectedRoof) -> float:
        """Calculate complexity factor for roof segment"""
        # Factors affecting complexity:
        # 1. Number of vertices
        num_vertices = len(detected_roof.polygon_points)
        
        # 2. Shape irregularity
        polygon = Polygon(detected_roof.polygon_points)
        if polygon.is_valid:
            # Calculate convex hull
            convex_hull = polygon.convex_hull
            # Complexity ratio
            complexity_ratio = polygon.area / convex_hull.area if convex_hull.area > 0 else 1.0
        else:
            complexity_ratio = 1.0
        
        # 3. Aspect ratio
        x, y, w, h = detected_roof.bounding_box
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
        
        # Calculate complexity factor
        vertex_factor = min(num_vertices / 6.0, 2.0)  # Normalize by typical rectangle (4 vertices)
        shape_factor = 2.0 - complexity_ratio  # Higher complexity for lower ratio
        aspect_factor = min(aspect_ratio / 2.0, 1.5)  # Higher complexity for extreme aspects
        
        complexity_factor = (vertex_factor + shape_factor + aspect_factor) / 3.0
        
        return min(max(complexity_factor, 1.0), 3.0)
    
    def _determine_orientation(self, polygon_points: List[Tuple[float, float]]) -> str:
        """Determine roof orientation"""
        if len(polygon_points) < 3:
            return "unknown"
        
        # Calculate principal axis using PCA
        points = np.array(polygon_points)
        centered = points - np.mean(points, axis=0)
        
        # Covariance matrix
        cov = np.cov(centered.T)
        
        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Principal axis
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
        angle = math.degrees(math.atan2(principal_axis[1], principal_axis[0]))
        
        # Normalize angle to 0-180 degrees
        angle = angle % 180
        
        # Determine orientation
        if 45 <= angle < 135:
            return "east_west"
        else:
            return "north_south"
    
    def _convert_coordinates(
        self,
        polygon_points: List[Tuple[float, float]],
        pixel_ratio: float,
        imagery_data: Dict[str, Any]
    ) -> List[Tuple[float, float]]:
        """Convert pixel coordinates to real-world coordinates"""
        # This is a simplified conversion
        # In practice, you would use proper georeferencing
        
        metadata = imagery_data.get('metadata', {})
        center_lat = metadata.get('coordinates', [0, 0])[0]
        center_lon = metadata.get('coordinates', [0, 0])[1]
        
        # Simple conversion (not accurate for real-world use)
        converted_points = []
        for px, py in polygon_points:
            # Convert pixels to feet
            ft_x = (px - 512) * pixel_ratio  # Assuming 1024x1024 image centered
            ft_y = (py - 512) * pixel_ratio
            
            # Convert feet to approximate degrees (very rough approximation)
            deg_lat = center_lat + (ft_y / 364000)  # 1 degree ≈ 364,000 ft
            deg_lon = center_lon + (ft_x / (364000 * math.cos(math.radians(center_lat))))
            
            converted_points.append((deg_lat, deg_lon))
        
        return converted_points
    
    def _calculate_segment_confidence(
        self,
        detected_roof: DetectedRoof,
        pitch_result: Dict[str, Any],
        area_sq_ft: float
    ) -> float:
        """Calculate confidence score for segment measurement"""
        
        # Base confidence from detection
        base_confidence = detected_roof.confidence_score
        
        # Pitch confidence
        pitch_confidence = pitch_result.get('confidence', 0.5)
        
        # Area confidence (larger areas are more reliable)
        area_confidence = min(area_sq_ft / 200, 1.0)  # Normalize to 200 sq ft
        
        # Measurement method confidence
        method_confidence = 1.0 if len(pitch_result.get('methods_used', [])) > 1 else 0.8
        
        # Combined confidence
        combined_confidence = (
            base_confidence * 0.3 +
            pitch_confidence * 0.3 +
            area_confidence * 0.2 +
            method_confidence * 0.2
        )
        
        return min(combined_confidence, 1.0)
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess image quality for measurement"""
        # Calculate various quality metrics
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Brightness (mean intensity)
        brightness = np.mean(gray)
        
        # Noise estimation
        noise = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))
        
        return {
            'sharpness': sharpness,
            'contrast': contrast,
            'brightness': brightness,
            'noise': noise,
            'overall_quality': min(sharpness / 100 + contrast / 50, 1.0)
        }
    
    def _calculate_roof_complexity(self, segments: List[RoofSegment]) -> float:
        """Calculate overall roof complexity"""
        if not segments:
            return 1.0
        
        # Average complexity factor
        avg_complexity = np.mean([seg.complexity_factor for seg in segments])
        
        # Number of segments factor
        segment_factor = min(len(segments) / 3.0, 2.0)
        
        # Area variation factor
        areas = [seg.area_sq_ft for seg in segments]
        area_variation = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
        
        # Combined complexity
        complexity = (avg_complexity + segment_factor + area_variation) / 3.0
        
        return min(max(complexity, 1.0), 3.0)
    
    def _estimate_building_height(self, segments: List[RoofSegment], avg_pitch: float) -> float:
        """Estimate building height from roof segments"""
        if not segments:
            return 10.0  # Default height
        
        # Estimate height based on roof pitch and area
        # Simplified calculation: height ≈ sqrt(area) * tan(pitch) / 2
        total_area = sum(seg.area_sq_ft for seg in segments)
        avg_dimension = math.sqrt(total_area)
        
        # Height from roof pitch
        roof_height = avg_dimension * math.tan(math.radians(avg_pitch)) / 2
        
        # Add base height (typical single story)
        base_height = 10.0
        
        total_height = base_height + roof_height
        
        return min(max(total_height, 8.0), 50.0)  # Reasonable bounds
    
    def _estimate_number_of_stories(self, height_ft: float) -> int:
        """Estimate number of stories from height"""
        # Typical story heights: 8-12 feet
        story_height = 10.0
        stories = round(height_ft / story_height)
        
        return max(stories, 1)  # At least 1 story
    
    def _calculate_accuracy_score(self, segments: List[RoofSegment], imagery_data: Dict[str, Any]) -> float:
        """Calculate overall measurement accuracy score"""
        if not segments:
            return 0.5
        
        # Individual segment confidences
        segment_confidences = []
        for seg in segments:
            seg_confidence = seg.features.get('detection_confidence', 0.5)
            pitch_confidence = seg.features.get('pitch_confidence', 0.5)
            area_confidence = seg.features.get('area_confidence', 0.5)
            
            combined = (seg_confidence + pitch_confidence + area_confidence) / 3.0
            segment_confidences.append(combined)
        
        avg_segment_confidence = np.mean(segment_confidences)
        
        # Image quality factor
        image_quality = imagery_data.get('metadata', {}).get('image_quality', {}).get('overall_quality', 0.7)
        
        # Number of segments factor (more segments can indicate better detection)
        segment_factor = min(len(segments) / 5.0, 1.0)
        
        # Overall accuracy
        accuracy = (
            avg_segment_confidence * 0.5 +
            image_quality * 0.3 +
            segment_factor * 0.2
        )
        
        return min(max(accuracy, 0.1), 1.0)
    
    def get_measurement_stats(self) -> Dict[str, Any]:
        """Get measurement statistics"""
        return {
            'pixel_to_foot_ratio': self.pixel_to_foot_ratio,
            'min_area_threshold': self.min_area_threshold,
            'default_pitch': self.default_pitch,
            'shadow_analysis_enabled': self.shadow_analysis_enabled,
            'stereo_analysis_enabled': self.stereo_analysis_enabled,
            'accuracy_tolerance': self.accuracy_tolerance,
            'confidence_threshold': self.confidence_threshold
        }