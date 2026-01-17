import pytesseract
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class OCRWord:
    """Data class for OCR word results"""
    text: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    page_num: int

@dataclass
class OCRResult:
    """Data class for complete OCR results"""
    page_num: int
    text: str
    words: List[OCRWord]
    average_confidence: float
    engine_used: str
    metadata: Dict[str, Any]

class HybridOCREngine:
    """Hybrid OCR Engine with Tesseract + optional PaddleOCR fallback"""
    
    def __init__(self):
        self.tesseract_config = '--oem 3 --psm 6'
        self.confidence_threshold = settings.OCR_CONFIDENCE_THRESHOLD
        self.paddle_ocr = None
        
        # Try to import PaddleOCR (optional)
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            logger.info("PaddleOCR loaded successfully (optional fallback)")
        except ImportError:
            logger.info("PaddleOCR not installed, using Tesseract only")
        except Exception as e:
            logger.warning(f"PaddleOCR initialization failed: {e}")
    
    def process_image(self, image: np.ndarray, page_num: int = 0) -> OCRResult:
        """
        Process image with OCR
        
        Args:
            image: Input image
            page_num: Page number
            
        Returns:
            OCRResult object
        """
        # Always use Tesseract first
        tesseract_result = self._process_with_tesseract(image, page_num)
        
        # Check if confidence is sufficient
        if tesseract_result.average_confidence >= self.confidence_threshold:
            logger.info(f"Tesseract confidence sufficient: {tesseract_result.average_confidence:.2f}")
            return tesseract_result
        
        # Fallback to PaddleOCR if available
        if self.paddle_ocr is not None:
            logger.info(f"Tesseract confidence low ({tesseract_result.average_confidence:.2f}), trying PaddleOCR")
            paddle_result = self._process_with_paddleocr(image, page_num)
            
            # Choose the result with higher confidence
            if paddle_result.average_confidence > tesseract_result.average_confidence:
                return paddle_result
        
        return tesseract_result
    
    def _process_with_tesseract(self, image: np.ndarray, page_num: int) -> OCRResult:
        """Process image using Tesseract OCR"""
        try:
            # Convert to PIL Image
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image)
            elif image.shape[2] == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif image.shape[2] == 4:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
            else:
                pil_image = Image.fromarray(image[:, :, :3])
            
            # Get data with bounding boxes
            data = pytesseract.image_to_data(
                pil_image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract words and confidence
            words = []
            confidences = []
            
            for i in range(len(data['text'])):
                if data['text'][i].strip():  # Non-empty text
                    word = OCRWord(
                        text=data['text'][i],
                        confidence=float(data['conf'][i]) / 100.0,
                        bbox=[
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ],
                        page_num=page_num
                    )
                    words.append(word)
                    confidences.append(word.confidence)
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Full text
            full_text = pytesseract.image_to_string(pil_image)
            
            return OCRResult(
                page_num=page_num,
                text=full_text,
                words=words,
                average_confidence=avg_confidence,
                engine_used="tesseract",
                metadata={
                    "total_words": len(words),
                    "min_confidence": min(confidences) if confidences else 0,
                    "max_confidence": max(confidences) if confidences else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return self._create_empty_result(page_num, "tesseract_failed")
    
    def _process_with_paddleocr(self, image: np.ndarray, page_num: int) -> OCRResult:
        """Process image using PaddleOCR (if available)"""
        try:
            if self.paddle_ocr is None:
                return self._create_empty_result(page_num, "paddleocr_unavailable")
            
            # Convert to RGB if needed
            if len(image.shape) == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                rgb_image = image
            
            # Run PaddleOCR
            result = self.paddle_ocr.ocr(rgb_image, cls=True)
            
            words = []
            confidences = []
            full_text_lines = []
            
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        bbox, (text, confidence) = line[0], line[1]
                        
                        # Flatten bbox
                        flat_bbox = [int(coord) for point in bbox for coord in point]
                        
                        word = OCRWord(
                            text=text,
                            confidence=confidence,
                            bbox=flat_bbox,
                            page_num=page_num
                        )
                        words.append(word)
                        confidences.append(confidence)
                        full_text_lines.append(text)
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return OCRResult(
                page_num=page_num,
                text='\n'.join(full_text_lines),
                words=words,
                average_confidence=avg_confidence,
                engine_used="paddleocr",
                metadata={
                    "total_words": len(words),
                    "min_confidence": min(confidences) if confidences else 0,
                    "max_confidence": max(confidences) if confidences else 0
                }
            )
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return self._create_empty_result(page_num, "paddleocr_failed")
    
    def _create_empty_result(self, page_num: int, engine: str) -> OCRResult:
        """Create empty result for failed OCR"""
        return OCRResult(
            page_num=page_num,
            text="",
            words=[],
            average_confidence=0.0,
            engine_used=engine,
            metadata={"error": "OCR processing failed"}
        )
    
    def process_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """Process multiple images"""
        results = []
        for idx, image in enumerate(images):
            result = self.process_image(image, page_num=idx)
            results.append(result)
        return results