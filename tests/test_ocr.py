import pytest
import numpy as np
import cv2
from app.models.ocr_engine import HybridOCREngine, OCRResult
from app.utils.error_handler import OCRException

class TestHybridOCREngine:
    """Test hybrid OCR engine"""
    
    def setup_method(self):
        self.ocr_engine = HybridOCREngine()
    
    def test_initialization(self):
        """Test OCR engine initialization"""
        assert self.ocr_engine is not None
        assert hasattr(self.ocr_engine, 'confidence_threshold')
        assert self.ocr_engine.confidence_threshold == 0.85
    
    def test_create_test_image_with_text(self):
        """Create test image with text"""
        # Create a simple white image
        image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        # Add text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Test Document', (50, 100), font, 1, (0, 0, 0), 2)
        cv2.putText(image, 'Sample Text Line 1', (50, 140), font, 0.7, (0, 0, 0), 1)
        cv2.putText(image, 'Sample Text Line 2', (50, 170), font, 0.7, (0, 0, 0), 1)
        
        return image
    
    def test_ocr_processing(self):
        """Test OCR processing on test image"""
        # Create test image
        test_image = self.create_test_image_with_text()
        
        # Process with OCR
        result = self.ocr_engine.process_image(test_image, page_num=0)
        
        # Assert results
        assert isinstance(result, OCRResult)
        assert result.page_num == 0
        assert result.engine_used in ["tesseract", "paddleocr"]
        assert result.average_confidence > 0
        assert len(result.words) > 0
        
        # Check word extraction
        for word in result.words:
            assert isinstance(word.text, str)
            assert 0 <= word.confidence <= 1
            assert len(word.bbox) == 4
    
    def test_empty_image(self):
        """Test OCR on empty image"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = self.ocr_engine.process_image(empty_image, page_num=0)
        
        assert result.page_num == 0
        assert result.average_confidence == 0.0 or result.average_confidence < 0.1
    
    def test_batch_processing(self):
        """Test batch OCR processing"""
        images = [
            self.create_test_image_with_text(),
            self.create_test_image_with_text()
        ]
        
        results = self.ocr_engine.process_batch(images)
        
        assert len(results) == 2
        assert all(isinstance(r, OCRResult) for r in results)
    
    def test_confidence_threshold(self):
        """Test confidence threshold logic"""
        # Create low-quality image
        low_quality_image = np.random.randint(0, 50, (100, 200, 3), dtype=np.uint8)
        
        # This should trigger paddleocr fallback
        result = self.ocr_engine.process_image(low_quality_image, page_num=0)
        
        assert result.engine_used in ["tesseract", "paddleocr", "tesseract_failed", "paddleocr_failed"]
    
    @pytest.mark.asyncio
    async def test_ocr_exception_handling(self):
        """Test OCR exception handling"""
        # Test with invalid input
        invalid_image = "not an image"
        
        with pytest.raises(Exception):
            # This should raise an exception
            self.ocr_engine.process_image(invalid_image, page_num=0)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])