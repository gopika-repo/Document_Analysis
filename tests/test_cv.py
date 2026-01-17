import pytest
import numpy as np
import cv2
from app.models.yolo_loader import YOLOModel, DetectionResult
from app.models.layout_analyzer import LayoutAnalyzer
from app.utils.error_handler import CVException

class TestYOLOModel:
    """Test YOLO model for document element detection"""
    
    def setup_method(self):
        self.yolo_model = YOLOModel()
    
    def create_test_document_image(self):
        """Create test document image with various elements"""
        # Create a white document-like image
        image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Add a table (black rectangle with grid)
        cv2.rectangle(image, (50, 100), (300, 300), (0, 0, 0), 2)
        for i in range(1, 5):
            cv2.line(image, (50, 100 + i*40), (300, 100 + i*40), (0, 0, 0), 1)
            cv2.line(image, (50 + i*50, 100), (50 + i*50, 300), (0, 0, 0), 1)
        
        # Add a chart (circle with lines)
        cv2.circle(image, (450, 200), 50, (255, 0, 0), -1)
        cv2.line(image, (400, 200), (500, 200), (0, 0, 0), 2)
        cv2.line(image, (450, 150), (450, 250), (0, 0, 0), 2)
        
        # Add text regions
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Document Title', (200, 50), font, 1, (0, 0, 0), 2)
        cv2.putText(image, 'Sample paragraph text for testing.', (50, 350), font, 0.5, (0, 0, 0), 1)
        
        # Add signature-like squiggle
        points = np.array([[400, 400], [420, 410], [440, 390], [460, 420], [480, 400]], np.int32)
        cv2.polylines(image, [points], False, (0, 0, 0), 2)
        
        return image
    
    def test_initialization(self):
        """Test YOLO model initialization"""
        assert self.yolo_model is not None
        assert hasattr(self.yolo_model, 'model')
        assert hasattr(self.yolo_model, 'confidence_threshold')
        assert self.yolo_model.confidence_threshold == 0.5
    
    def test_detection(self):
        """Test element detection"""
        test_image = self.create_test_document_image()
        
        detections = self.yolo_model.detect(test_image, page_num=0)
        
        assert isinstance(detections, list)
        
        # Check detection results
        for detection in detections:
            assert isinstance(detection, DetectionResult)
            assert detection.class_name in self.yolo_model.classes
            assert 0 <= detection.confidence <= 1
            assert len(detection.bbox) == 4
            assert detection.page_num == 0
    
    def test_batch_detection(self):
        """Test batch detection"""
        images = [
            self.create_test_document_image(),
            self.create_test_document_image()
        ]
        
        results = self.yolo_model.batch_detect(images)
        
        assert isinstance(results, dict)
        assert len(results) == 2
        assert all(isinstance(dets, list) for dets in results.values())
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        test_image = self.create_test_document_image()
        
        # Test region feature extraction
        region = test_image[100:200, 100:200]
        features = self.yolo_model._extract_features(region)
        
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_empty_image(self):
        """Test detection on empty image"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        detections = self.yolo_model.detect(empty_image, page_num=0)
        
        assert isinstance(detections, list)
        # May have zero detections on empty image
    
    @pytest.mark.asyncio
    async def test_cv_exception_handling(self):
        """Test CV exception handling"""
        # Test with invalid input
        invalid_image = None
        
        with pytest.raises(Exception):
            self.yolo_model.detect(invalid_image, page_num=0)

class TestLayoutAnalyzer:
    """Test layout analyzer"""
    
    def setup_method(self):
        self.layout_analyzer = LayoutAnalyzer()
    
    def test_initialization(self):
        """Test layout analyzer initialization"""
        assert self.layout_analyzer is not None
        assert hasattr(self.layout_analyzer, 'header_threshold')
        assert hasattr(self.layout_analyzer, 'footer_threshold')
    
    def test_layout_analysis(self):
        """Test layout analysis"""
        test_image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        
        # Add some text regions
        cv2.rectangle(test_image, (100, 50), (700, 150), (200, 200, 200), -1)  # Header
        cv2.rectangle(test_image, (100, 200), (700, 800), (240, 240, 240), -1)  # Body
        cv2.rectangle(test_image, (100, 850), (700, 950), (200, 200, 200), -1)  # Footer
        
        layout = self.layout_analyzer.analyze(test_image, page_num=0)
        
        assert isinstance(layout, dict)
        assert "page_number" in layout
        assert "regions" in layout
        assert "spatial_relationships" in layout
        
        # Check region classification
        regions = layout.get("regions", [])
        assert len(regions) > 0
        
        # Check region types
        region_types = [r.get("type") for r in regions]
        assert any(rt in ["header", "body", "footer", "text_region"] for rt in region_types)
    
    def test_region_detection(self):
        """Test text region detection"""
        test_image = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # Add a white text region
        test_image[50:100, 50:150] = 255
        
        regions = self.layout_analyzer._detect_text_regions(test_image)
        
        assert isinstance(regions, list)
        assert len(regions) > 0
        
        for region in regions:
            assert len(region) == 4  # [x1, y1, x2, y2]
    
    def test_region_classification(self):
        """Test region classification"""
        # Create test regions
        test_regions = [
            type('Region', (), {'bbox': [100, 50, 700, 150], 'region_type': 'text_region'})(),
            type('Region', (), {'bbox': [100, 200, 700, 800], 'region_type': 'text_region'})()
        ]
        
        width, height = 800, 1000
        classified = self.layout_analyzer._classify_regions(test_regions, width, height)
        
        # Check classification
        for region in classified:
            center_y = (region.bbox[1] + region.bbox[3]) / 2
            
            if center_y < height * self.layout_analyzer.header_threshold:
                assert region.region_type == "header"
            elif center_y > height * self.layout_analyzer.footer_threshold:
                assert region.region_type == "footer"
            else:
                assert region.region_type in ["body", "sidebar"]
    
    def test_spatial_graph(self):
        """Test spatial graph building"""
        test_regions = [
            type('Region', (), {
                'bbox': [100, 100, 200, 200],
                'region_type': 'text_region'
            })(),
            type('Region', (), {
                'bbox': [210, 100, 310, 200],  # Adjacent horizontally
                'region_type': 'text_region'
            })()
        ]
        
        graph = self.layout_analyzer._build_spatial_graph(test_regions)
        
        assert isinstance(graph, dict)
        assert len(graph) == 2
        
        # Check adjacency
        for neighbors in graph.values():
            assert isinstance(neighbors, list)
    
    def test_empty_image_analysis(self):
        """Test layout analysis on empty image"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        layout = self.layout_analyzer.analyze(empty_image, page_num=0)
        
        assert isinstance(layout, dict)
        assert layout["page_number"] == 0
        assert len(layout.get("regions", [])) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])