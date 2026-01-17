import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from app.core.models import ProcessingState, VisualElement
from app.utils.logger import setup_logger
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

logger = setup_logger(__name__)

class ChartUnderstandingAgent:
    """Agent for understanding and interpreting charts"""
    
    def __init__(self):
        self.trend_keywords = ["increase", "decrease", "growth", "decline", "steady", "fluctuate"]
    
    async def analyze_charts(self, state: ProcessingState) -> ProcessingState:
        """Analyze detected charts in the document"""
        try:
            logger.info(f"Analyzing charts for {state.document_id}")
            
            chart_analysis = {}
            
            # Find chart elements
            charts = self._extract_chart_elements(state.visual_elements)
            
            for chart_id, (page_num, bbox, image) in enumerate(charts):
                analysis = self._analyze_single_chart(image, bbox, page_num)
                chart_analysis[f"chart_{chart_id}"] = analysis
            
            state.chart_analysis = chart_analysis
            logger.info(f"Analyzed {len(chart_analysis)} charts")
            
            return state
            
        except Exception as e:
            logger.error(f"Chart analysis failed: {e}")
            state.errors.append(f"Chart analysis error: {str(e)}")
            return state
    
    def _extract_chart_elements(self, visual_elements: Dict[int, List[VisualElement]]) -> List[tuple]:
        """Extract chart elements from visual detections"""
        charts = []
        
        for page_num, elements in visual_elements.items():
            for element in elements:
                if element.element_type == "chart":
                    # In production, this would extract the actual image region
                    # For now, return metadata
                    charts.append((
                        page_num,
                        element.bbox,
                        element.metadata.get("image_region", None)
                    ))
        
        return charts
    
    def _analyze_single_chart(self, image: Optional[np.ndarray], 
                             bbox: List[int], page_num: int) -> Dict[str, Any]:
        """Analyze a single chart image"""
        analysis = {
            "page": page_num,
            "bbox": bbox,
            "chart_type": "unknown",
            "trend_direction": "unknown",
            "data_points": [],
            "confidence": 0.0,
            "features": {}
        }
        
        if image is None:
            return analysis
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect chart type based on features
            chart_type = self._detect_chart_type(gray)
            analysis["chart_type"] = chart_type
            
            # Extract data points (simplified)
            if chart_type in ["bar", "line"]:
                data_points = self._extract_data_points(gray, chart_type)
                analysis["data_points"] = data_points
                
                # Determine trend
                if len(data_points) > 1:
                    trend = self._determine_trend(data_points)
                    analysis["trend_direction"] = trend
            
            # Calculate confidence based on clarity
            analysis["confidence"] = self._calculate_chart_confidence(gray)
            
            # Extract features
            analysis["features"] = {
                "clarity": self._calculate_clarity(gray),
                "complexity": self._calculate_complexity(gray),
                "axis_presence": self._detect_axes(gray)
            }
            
        except Exception as e:
            logger.warning(f"Single chart analysis failed: {e}")
        
        return analysis
    
    def _detect_chart_type(self, image: np.ndarray) -> str:
        """Detect type of chart"""
        try:
            # Edge detection
            edges = cv2.Canny(image, 50, 150)
            
            # Detect lines for line charts
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            
            # Detect circles for pie charts
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                                      param1=50, param2=30, minRadius=10, maxRadius=100)
            
            # Detect rectangles for bar charts
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = [cnt for cnt in contours if len(cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)) == 4]
            
            # Determine type
            if circles is not None and len(circles[0]) > 3:
                return "pie"
            elif lines is not None and len(lines) > 5:
                return "line"
            elif len(rectangles) > 3:
                return "bar"
            else:
                return "unknown"
                
        except Exception as e:
            logger.warning(f"Chart type detection failed: {e}")
            return "unknown"
    
    def _extract_data_points(self, image: np.ndarray, chart_type: str) -> List[float]:
        """Extract data points from chart (simplified)"""
        # In production, this would use proper computer vision techniques
        # For demo, return simulated data
        if chart_type == "bar":
            return [0.3, 0.5, 0.7, 0.9, 0.8]  # Simulated bar heights
        elif chart_type == "line":
            return [0.2, 0.4, 0.6, 0.8, 0.7]  # Simulated line points
        else:
            return []
    
    def _determine_trend(self, data_points: List[float]) -> str:
        """Determine trend direction from data points"""
        if len(data_points) < 2:
            return "unknown"
        
        # Calculate slope
        x = np.arange(len(data_points))
        y = np.array(data_points)
        
        if len(y) > 1:
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.1:
                return "increasing"
            elif slope < -0.1:
                return "decreasing"
            else:
                return "stable"
        
        return "unknown"