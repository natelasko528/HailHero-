"""
Roof detection algorithms using computer vision
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import io
import logging
from dataclasses import dataclass
import uuid
from scipy import ndimage
from sklearn.cluster import KMeans
import math


@dataclass
class DetectedRoof:
    """Detected roof segment with coordinates and properties"""
    segment_id: str
    roof_type: str
    polygon_points: List[Tuple[float, float]]
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    area_pixels: int
    confidence_score: float
    color_features: Dict[str, Any]
    texture_features: Dict[str, Any]
    shape_features: Dict[str, Any]
    
    def to_area_sq_ft(self, pixel_to_sq_ft_ratio: float) -> float:
        """Convert pixel area to square feet"""
        return self.area_pixels * pixel_to_sq_ft_ratio


class RoofDetector:
    """Roof detection using computer vision algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize roof detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters
        self.roof_colors = config.get('roof_colors', [
            [100, 100, 100],   # Gray
            [150, 75, 0],      # Brown
            [200, 150, 100],   # Tan
            [50, 50, 50],      # Dark gray
            [255, 255, 255],   # White
            [100, 150, 200],   # Blue-gray
        ])
        
        self.min_roof_area = config.get('min_roof_area', 1000)  # pixels
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.roof_type_models = config.get('roof_type_models', {})
        
        # Initialize models
        self._initialize_models()
        
        self.logger.info("RoofDetector initialized")
    
    def _initialize_models(self):
        """Initialize ML models for roof detection"""
        try:
            # Initialize color analysis models
            self.color_analyzer = self._initialize_color_analyzer()
            
            # Initialize texture analysis models
            self.texture_analyzer = self._initialize_texture_analyzer()
            
            # Initialize shape classification models
            self.shape_classifier = self._initialize_shape_classifier()
            
            # Initialize damage detection models
            self.damage_detector = self._initialize_damage_detector()
            
            # Initialize feature extractors
            self.feature_extractors = self._initialize_feature_extractors()
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            # Fall back to traditional computer vision methods
            self._initialize_fallback_methods()
    
    def _initialize_color_analyzer(self):
        """Initialize color analysis components"""
        return {
            'kmeans_clusterer': KMeans(n_clusters=5, random_state=42, n_init=10),
            'color_histogram_bins': 32,
            'color_space_weights': {
                'hsv': 0.4,
                'lab': 0.3,
                'rgb': 0.3
            }
        }
    
    def _initialize_texture_analyzer(self):
        """Initialize texture analysis components"""
        return {
            'lbp_radius': 3,
            'lbp_neighbors': 8,
            'glcm_distances': [1, 2, 3],
            'glcm_angles': [0, 45, 90, 135],
            'texture_thresholds': {
                'smooth': 0.1,
                'rough': 0.3,
                'patterned': 0.5
            }
        }
    
    def _initialize_shape_classifier(self):
        """Initialize shape classification components"""
        return {
            'shape_templates': {
                'gable': self._create_gable_template(),
                'hip': self._create_hip_template(),
                'flat': self._create_flat_template(),
                'mansard': self._create_mansard_template(),
                'gambrel': self._create_gambrel_template()
            },
            'shape_matching_threshold': 0.7,
            'aspect_ratio_ranges': {
                'gable': (1.2, 3.0),
                'hip': (0.8, 1.5),
                'flat': (0.5, 2.0),
                'mansard': (0.8, 2.0),
                'gambrel': (1.0, 2.5)
            }
        }
    
    def _initialize_damage_detector(self):
        """Initialize damage detection components"""
        return {
            'anomaly_threshold': 2.0,
            'damage_patterns': {
                'hail_dents': self._create_hail_dent_detector(),
                'cracks': self._create_crack_detector(),
                'missing_materials': self._create_missing_material_detector(),
                'discoloration': self._create_discoloration_detector()
            },
            'severity_levels': {
                'none': 0.0,
                'minor': 0.3,
                'moderate': 0.6,
                'severe': 0.9
            }
        }
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction components"""
        return {
            'color_features': ['mean', 'std', 'dominant', 'histogram'],
            'texture_features': ['lbp', 'glcm', 'edge_density', 'contrast'],
            'shape_features': ['area', 'perimeter', 'circularity', 'convexity', 'aspect_ratio'],
            'structural_features': ['line_density', 'corner_density', 'symmetry']
        }
    
    def _initialize_fallback_methods(self):
        """Initialize fallback traditional computer vision methods"""
        self.color_analyzer = {
            'kmeans_clusterer': KMeans(n_clusters=3, random_state=42, n_init=10),
            'color_histogram_bins': 16,
            'color_space_weights': {'rgb': 1.0}
        }
        
        self.texture_analyzer = {
            'lbp_radius': 1,
            'lbp_neighbors': 8,
            'texture_thresholds': {'smooth': 0.2, 'rough': 0.5}
        }
        
        self.shape_classifier = {
            'shape_templates': {
                'gable': self._create_gable_template(),
                'hip': self._create_hip_template(),
                'flat': self._create_flat_template()
            },
            'shape_matching_threshold': 0.6
        }
        
        self.damage_detector = {
            'anomaly_threshold': 1.5,
            'damage_patterns': {
                'hail_dents': self._create_hail_dent_detector()
            },
            'severity_levels': {'none': 0.0, 'minor': 0.4, 'moderate': 0.7, 'severe': 0.9}
        }
        
        self.feature_extractors = {
            'color_features': ['mean', 'std'],
            'texture_features': ['lbp', 'edge_density'],
            'shape_features': ['area', 'aspect_ratio']
        }
    
    def _create_gable_template(self):
        """Create template for gable roof detection"""
        return {
            'expected_vertices': 4,
            'peak_detection': True,
            'symmetry_requirement': 0.7,
            'slope_range': (0.3, 0.8)
        }
    
    def _create_hip_template(self):
        """Create template for hip roof detection"""
        return {
            'expected_vertices': [4, 6],
            'peak_detection': False,
            'symmetry_requirement': 0.6,
            'slope_range': (0.2, 0.6)
        }
    
    def _create_flat_template(self):
        """Create template for flat roof detection"""
        return {
            'expected_vertices': [4, 6, 8],
            'peak_detection': False,
            'symmetry_requirement': 0.5,
            'slope_range': (0.0, 0.2)
        }
    
    def _create_mansard_template(self):
        """Create template for mansard roof detection"""
        return {
            'expected_vertices': [6, 8],
            'peak_detection': True,
            'symmetry_requirement': 0.8,
            'slope_range': (0.4, 0.9)
        }
    
    def _create_gambrel_template(self):
        """Create template for gambrel roof detection"""
        return {
            'expected_vertices': [6, 8],
            'peak_detection': True,
            'symmetry_requirement': 0.7,
            'slope_range': (0.3, 0.7)
        }
    
    def _create_hail_dent_detector(self):
        """Create hail dent detection parameters"""
        return {
            'min_dent_size': 5,
            'max_dent_size': 50,
            'circularity_threshold': 0.6,
            'contrast_threshold': 30,
            'density_threshold': 0.1
        }
    
    def _create_crack_detector(self):
        """Create crack detection parameters"""
        return {
            'min_crack_length': 20,
            'max_crack_width': 5,
            'linearity_threshold': 0.8,
            'contrast_threshold': 40
        }
    
    def _create_missing_material_detector(self):
        """Create missing material detection parameters"""
        return {
            'min_hole_size': 100,
            'color_deviation_threshold': 50,
            'texture_disruption_threshold': 0.3
        }
    
    def _create_discoloration_detector(self):
        """Create discoloration detection parameters"""
        return {
            'color_shift_threshold': 30,
            'area_threshold': 0.05,
            'pattern_consistency': 0.6
        }
    
    async def detect_roofs(
        self,
        imagery_data: Dict[str, Any],
        property_location: Any
    ) -> List[DetectedRoof]:
        """
        Detect roof segments in aerial imagery
        
        Args:
            imagery_data: Imagery data from provider
            property_location: Property location information
            
        Returns:
            List of detected roof segments
        """
        try:
            # Extract image from imagery data
            image = self._extract_image(imagery_data)
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Detect roof candidates
            roof_candidates = self._detect_roof_candidates(processed_image)
            
            # Classify roof types
            classified_roofs = self._classify_roof_types(roof_candidates, processed_image)
            
            # Filter and validate
            valid_roofs = self._filter_and_validate(classified_roofs)
            
            self.logger.info(f"Detected {len(valid_roofs)} roof segments")
            return valid_roofs
            
        except Exception as e:
            self.logger.error(f"Error in roof detection: {str(e)}")
            raise
    
    def _extract_image(self, imagery_data: Dict[str, Any]) -> np.ndarray:
        """Extract image array from imagery data"""
        image_data = imagery_data['image_data']
        
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert to RGB if necessary
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        return image_array
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for roof detection"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Enhance contrast
        enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        return {
            'original': image,
            'hsv': hsv,
            'lab': lab,
            'enhanced': enhanced,
            'blurred': blurred
        }
    
    def _detect_roof_candidates(self, processed_images: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect potential roof areas"""
        candidates = []
        
        # Method 1: Color-based segmentation
        color_segments = self._color_based_segmentation(processed_images)
        candidates.extend(color_segments)
        
        # Method 2: Edge detection
        edge_segments = self._edge_based_detection(processed_images)
        candidates.extend(edge_segments)
        
        # Method 3: Texture analysis
        texture_segments = self._texture_based_detection(processed_images)
        candidates.extend(texture_segments)
        
        # Method 4: Building footprint detection
        footprint_segments = self._building_footprint_detection(processed_images)
        candidates.extend(footprint_segments)
        
        return candidates
    
    def _color_based_segmentation(self, processed_images: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect roofs using color segmentation"""
        segments = []
        image = processed_images['original']
        hsv = processed_images['hsv']
        
        # Convert roof colors to HSV ranges
        for i, roof_color in enumerate(self.roof_colors):
            rgb_color = np.array(roof_color, dtype=np.uint8)
            hsv_color = cv2.cvtColor(np.array([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]
            
            # Define HSV range
            lower_bound = np.array([
                max(0, hsv_color[0] - 20),
                max(0, hsv_color[1] - 50),
                max(0, hsv_color[2] - 50)
            ])
            upper_bound = np.array([
                min(179, hsv_color[0] + 20),
                min(255, hsv_color[1] + 50),
                min(255, hsv_color[2] + 50)
            ])
            
            # Create mask
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_roof_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Approximate polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Convert to polygon points
                    polygon_points = [(int(point[0][0]), int(point[0][1])) for point in approx]
                    
                    segments.append({
                        'contour': contour,
                        'polygon_points': polygon_points,
                        'bounding_box': (x, y, w, h),
                        'area': area,
                        'method': 'color',
                        'color_index': i,
                        'confidence': 0.8
                    })
        
        return segments
    
    def _edge_based_detection(self, processed_images: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect roofs using edge detection"""
        segments = []
        image = processed_images['blurred']
        
        # Apply Canny edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_roof_area:
                # Check if contour is roughly rectangular
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio
                    # Approximate polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    polygon_points = [(int(point[0][0]), int(point[0][1])) for point in approx]
                    
                    segments.append({
                        'contour': contour,
                        'polygon_points': polygon_points,
                        'bounding_box': (x, y, w, h),
                        'area': area,
                        'method': 'edge',
                        'confidence': 0.6
                    })
        
        return segments
    
    def _texture_based_detection(self, processed_images: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect roofs using texture analysis"""
        segments = []
        gray = cv2.cvtColor(processed_images['original'], cv2.COLOR_RGB2GRAY)
        
        # Calculate local binary pattern (simplified)
        kernel_size = 3
        lbp = np.zeros_like(gray)
        
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                center = gray[i, j]
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                binary = sum([1 if n >= center else 0 for n in neighbors])
                lbp[i, j] = binary
        
        # Normalize LBP
        lbp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Threshold to find uniform texture regions
        _, binary = cv2.threshold(lbp, 100, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_roof_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check texture uniformity
                roi = lbp[y:y+h, x:x+w]
                texture_variance = np.var(roi)
                
                if texture_variance < 1000:  # Uniform texture
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    polygon_points = [(int(point[0][0]), int(point[0][1])) for point in approx]
                    
                    segments.append({
                        'contour': contour,
                        'polygon_points': polygon_points,
                        'bounding_box': (x, y, w, h),
                        'area': area,
                        'method': 'texture',
                        'confidence': 0.5
                    })
        
        return segments
    
    def _building_footprint_detection(self, processed_images: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect building footprints using structural analysis"""
        segments = []
        image = processed_images['original']
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Find structural lines
        lines = cv2.HoughLinesP(
            binary, 1, np.pi/180, threshold=50, 
            minLineLength=100, maxLineGap=10
        )
        
        if lines is not None:
            # Create line mask
            line_mask = np.zeros_like(gray)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            
            # Find intersections and create rectangular regions
            contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_roof_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it's a reasonable building shape
                    if w > 50 and h > 50:
                        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                        polygon_points = [(int(point[0][0]), int(point[0][1])) for point in approx]
                        
                        segments.append({
                            'contour': contour,
                            'polygon_points': polygon_points,
                            'bounding_box': (x, y, w, h),
                            'area': area,
                            'method': 'footprint',
                            'confidence': 0.7
                        })
        
        return segments
    
    def _classify_roof_types(self, candidates: List[Dict[str, Any]], processed_images: Dict[str, np.ndarray]) -> List[DetectedRoof]:
        """Classify roof types for detected segments"""
        classified_roofs = []
        
        for candidate in candidates:
            # Extract features
            features = self._extract_roof_features(candidate, processed_images)
            
            # Classify roof type
            roof_type = self._classify_roof_type(features)
            
            # Calculate confidence
            confidence = self._calculate_classification_confidence(features, roof_type)
            
            if confidence > self.confidence_threshold:
                detected_roof = DetectedRoof(
                    segment_id=f"roof_{uuid.uuid4().hex[:8]}",
                    roof_type=roof_type,
                    polygon_points=candidate['polygon_points'],
                    bounding_box=candidate['bounding_box'],
                    area_pixels=candidate['area'],
                    confidence_score=confidence,
                    color_features=features['color'],
                    texture_features=features['texture'],
                    shape_features=features['shape']
                )
                classified_roofs.append(detected_roof)
        
        return classified_roofs
    
    def _extract_roof_features(self, candidate: Dict[str, Any], processed_images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract features for roof classification"""
        x, y, w, h = candidate['bounding_box']
        image = processed_images['original']
        
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        
        # Color features
        color_features = {
            'mean_color': np.mean(roi, axis=(0, 1)),
            'std_color': np.std(roi, axis=(0, 1)),
            'dominant_color': self._get_dominant_color(roi)
        }
        
        # Texture features
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        texture_features = {
            'lbp_variance': np.var(gray_roi),
            'edge_density': self._calculate_edge_density(gray_roi),
            'contrast': self._calculate_contrast(gray_roi)
        }
        
        # Shape features
        contour = candidate['contour']
        shape_features = {
            'aspect_ratio': w / h if h > 0 else 1.0,
            'circularity': self._calculate_circularity(contour),
            'convexity': self._calculate_convexity(contour),
            'num_vertices': len(candidate['polygon_points'])
        }
        
        return {
            'color': color_features,
            'texture': texture_features,
            'shape': shape_features
        }
    
    def _get_dominant_color(self, image: np.ndarray) -> np.ndarray:
        """Get dominant color using k-means clustering"""
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    
    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """Calculate edge density in ROI"""
        edges = cv2.Canny(image, 50, 150)
        return np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate contrast using standard deviation"""
        return np.std(image)
    
    def _calculate_circularity(self, contour: np.ndarray) -> float:
        """Calculate circularity of contour"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            return 4 * np.pi * area / (perimeter ** 2)
        return 0
    
    def _calculate_convexity(self, contour: np.ndarray) -> float:
        """Calculate convexity of contour"""
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            return area / hull_area
        return 0
    
    def _classify_roof_type(self, features: Dict[str, Any]) -> str:
        """Classify roof type based on features"""
        shape = features['shape']
        
        # Simple rule-based classification
        num_vertices = shape['num_vertices']
        aspect_ratio = shape['aspect_ratio']
        circularity = shape['circularity']
        
        if num_vertices <= 4:
            if aspect_ratio > 1.5:
                return "gable"
            else:
                return "hip"
        elif num_vertices <= 6:
            return "complex"
        elif circularity > 0.7:
            return "flat"
        else:
            return "complex"
    
    def _calculate_classification_confidence(self, features: Dict[str, Any], roof_type: str) -> float:
        """Calculate confidence score for classification"""
        # Simple confidence calculation based on feature consistency
        shape = features['shape']
        
        if roof_type == "gable":
            if shape['num_vertices'] <= 4 and shape['aspect_ratio'] > 1.5:
                return 0.9
        elif roof_type == "hip":
            if shape['num_vertices'] <= 4 and shape['aspect_ratio'] <= 1.5:
                return 0.85
        elif roof_type == "flat":
            if shape['circularity'] > 0.7:
                return 0.8
        
        return 0.6  # Default confidence
    
    def _filter_and_validate(self, detected_roofs: List[DetectedRoof]) -> List[DetectedRoof]:
        """Filter and validate detected roofs"""
        valid_roofs = []
        
        for roof in detected_roofs:
            # Check minimum area
            if roof.area_pixels < self.min_roof_area:
                continue
            
            # Check confidence
            if roof.confidence_score < self.confidence_threshold:
                continue
            
            # Check shape validity
            if len(roof.polygon_points) < 3:
                continue
            
            valid_roofs.append(roof)
        
        return valid_roofs
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            'min_roof_area': self.min_roof_area,
            'confidence_threshold': self.confidence_threshold,
            'roof_colors_count': len(self.roof_colors),
            'methods_used': ['color', 'edge', 'texture', 'footprint']
        }