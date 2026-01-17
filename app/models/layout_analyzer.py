import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class LayoutRegion:
    """Data class for layout regions"""
    region_type: str  # header, body, footer, sidebar, table, figure
    bbox: List[int]   # [x1, y1, x2, y2]
    confidence: float
    page_num: int
    content: Optional[str] = None
    children: List[Any] = None

class LayoutAnalyzer:
    """Document Layout Analysis using Computer Vision"""
    
    def __init__(self):
        self.header_threshold = 0.15  # Top 15% of page
        self.footer_threshold = 0.85  # Bottom 15% of page
        self.sidebar_threshold = 0.2   # Left/Right 20% of page
        self.min_region_area = 1000    # Minimum area for regions
        
    def analyze(self, image: np.ndarray, 
                detections: List[Any] = None,
                page_num: int = 0) -> Dict[str, Any]:
        """
        Analyze document layout
        
        Args:
            image: Input image
            detections: Pre-existing detections (from YOLO)
            page_num: Page number
            
        Returns:
            Layout analysis results
        """
        try:
            height, width = image.shape[:2]
            
            # Initialize regions
            regions = []
            
            # Detect text regions using contours
            text_regions = self._detect_text_regions(image)
            for bbox in text_regions:
                region = LayoutRegion(
                    region_type="text_region",
                    bbox=bbox,
                    confidence=0.8,
                    page_num=page_num
                )
                regions.append(region)
            
            # Classify regions based on position
            classified_regions = self._classify_regions(regions, width, height)
            
            # Merge with YOLO detections if provided
            if detections:
                classified_regions = self._merge_with_detections(
                    classified_regions, detections, page_num
                )
            
            # Build spatial relationships
            spatial_graph = self._build_spatial_graph(classified_regions)
            
            # Generate layout JSON
            layout_json = self._generate_layout_json(
                classified_regions, spatial_graph, width, height, page_num
            )
            
            return layout_json
            
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            return self._create_empty_layout(page_num)
    
    def _detect_text_regions(self, image: np.ndarray) -> List[List[int]]:
        """Detect text regions using contour analysis"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_region_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append([x, y, x + w, y + h])
            
            # Merge overlapping regions
            merged_regions = self._merge_overlapping_regions(regions)
            
            return merged_regions
            
        except Exception as e:
            logger.error(f"Text region detection failed: {e}")
            return []
    
    def _merge_overlapping_regions(self, regions: List[List[int]], 
                                  overlap_threshold: float = 0.5) -> List[List[int]]:
        """Merge overlapping regions"""
        if not regions:
            return []
        
        regions = sorted(regions, key=lambda r: r[1])  # Sort by y-coordinate
        merged = []
        
        current = regions[0]
        for region in regions[1:]:
            # Calculate overlap
            x_overlap = max(0, min(current[2], region[2]) - max(current[0], region[0]))
            y_overlap = max(0, min(current[3], region[3]) - max(current[1], region[1]))
            
            overlap_area = x_overlap * y_overlap
            current_area = (current[2] - current[0]) * (current[3] - current[1])
            
            if overlap_area / current_area > overlap_threshold:
                # Merge regions
                current = [
                    min(current[0], region[0]),
                    min(current[1], region[1]),
                    max(current[2], region[2]),
                    max(current[3], region[3])
                ]
            else:
                merged.append(current)
                current = region
        
        merged.append(current)
        return merged
    
    def _classify_regions(self, regions: List[LayoutRegion], 
                         width: int, height: int) -> List[LayoutRegion]:
        """Classify regions based on position"""
        for region in regions:
            center_y = (region.bbox[1] + region.bbox[3]) / 2
            center_x = (region.bbox[0] + region.bbox[2]) / 2
            
            # Classify based on position
            if center_y < height * self.header_threshold:
                region.region_type = "header"
            elif center_y > height * self.footer_threshold:
                region.region_type = "footer"
            elif center_x < width * self.sidebar_threshold:
                region.region_type = "sidebar"
            elif center_x > width * (1 - self.sidebar_threshold):
                region.region_type = "sidebar"
            else:
                region.region_type = "body"
        
        return regions
    
    def _merge_with_detections(self, regions: List[LayoutRegion],
                              detections: List[Any],
                              page_num: int) -> List[LayoutRegion]:
        """Merge layout regions with YOLO detections"""
        for detection in detections:
            # Create region from detection
            region = LayoutRegion(
                region_type=detection.class_name,
                bbox=detection.bbox,
                confidence=detection.confidence,
                page_num=page_num
            )
            regions.append(region)
        
        return regions
    
    def _build_spatial_graph(self, regions: List[LayoutRegion]) -> Dict[str, List[str]]:
        """Build spatial relationship graph"""
        graph = {}
        
        for i, region in enumerate(regions):
            neighbors = []
            region_id = f"{region.region_type}_{i}"
            
            for j, other in enumerate(regions):
                if i != j:
                    # Check spatial relationships
                    if self._is_adjacent(region.bbox, other.bbox):
                        other_id = f"{other.region_type}_{j}"
                        neighbors.append(other_id)
            
            graph[region_id] = neighbors
        
        return graph
    
    def _is_adjacent(self, bbox1: List[int], bbox2: List[int], 
                    threshold: int = 20) -> bool:
        """Check if two regions are adjacent"""
        # Calculate horizontal and vertical distances
        horizontal_dist = max(0, 
                            max(bbox1[0], bbox2[0]) - min(bbox1[2], bbox2[2]))
        vertical_dist = max(0,
                          max(bbox1[1], bbox2[1]) - min(bbox1[3], bbox2[3]))
        
        return horizontal_dist < threshold or vertical_dist < threshold
    
    def _generate_layout_json(self, regions: List[LayoutRegion],
                            spatial_graph: Dict[str, List[str]],
                            width: int, height: int,
                            page_num: int) -> Dict[str, Any]:
        """Generate JSON representation of layout"""
        layout = {
            "page_number": page_num,
            "page_dimensions": {
                "width": width,
                "height": height
            },
            "regions": [],
            "spatial_relationships": spatial_graph,
            "metadata": {
                "total_regions": len(regions),
                "region_types": {}
            }
        }
        
        # Count region types
        type_counts = {}
        for region in regions:
            type_counts[region.region_type] = type_counts.get(region.region_type, 0) + 1
            
            # Add region to layout
            region_data = {
                "type": region.region_type,
                "bbox": region.bbox,
                "confidence": region.confidence,
                "area": (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
            }
            layout["regions"].append(region_data)
        
        layout["metadata"]["region_types"] = type_counts
        
        return layout
    
    def _create_empty_layout(self, page_num: int) -> Dict[str, Any]:
        """Create empty layout for failed analysis"""
        return {
            "page_number": page_num,
            "page_dimensions": {"width": 0, "height": 0},
            "regions": [],
            "spatial_relationships": {},
            "metadata": {
                "total_regions": 0,
                "region_types": {},
                "error": "Layout analysis failed"
            }
        }