from typing import Dict, List, Any
import re
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class OCRReliabilityAgent:
    """Agent for assessing OCR reliability and confidence"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self.confusing_characters = {
            '0': ['O', 'o'],
            '1': ['l', 'I'],
            '5': ['S', 's'],
            '8': ['B'],
            'O': ['0'],
            'l': ['1', 'I'],
            'I': ['1', 'l']
        }
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Assess OCR reliability"""
        try:
            logger.info(f"Assessing OCR reliability for {state.document_id}")
            
            ocr_confidence = {}
            ocr_issues = {}
            
            if hasattr(state, 'ocr_results') and state.ocr_results:
                for page_num, ocr_result in state.ocr_results.items():
                    page_confidence, page_issues = self._assess_page_ocr(ocr_result, page_num)
                    ocr_confidence[page_num] = page_confidence
                    if page_issues:
                        ocr_issues[page_num] = page_issues
            
            state.ocr_confidence = ocr_confidence
            state.processing_metadata = state.processing_metadata or {}
            state.processing_metadata["ocr_issues"] = ocr_issues
            
            # Calculate overall OCR confidence
            if ocr_confidence:
                avg_confidence = sum(ocr_confidence.values()) / len(ocr_confidence)
                state.processing_metadata["overall_ocr_confidence"] = avg_confidence
                logger.info(f"Overall OCR confidence: {avg_confidence:.2f}")
            
            return state
            
        except Exception as e:
            logger.error(f"OCR reliability assessment failed: {e}")
            state.ocr_confidence = {}
            return state
    
    def _assess_page_ocr(self, ocr_result: Any, page_num: int) -> tuple:
        """Assess OCR reliability for a single page"""
        confidence = 0.7  # Default confidence
        issues = []
        
        try:
            if isinstance(ocr_result, dict):
                # Extract text and confidence from OCR result
                text = ocr_result.get('text', '')
                avg_conf = ocr_result.get('average_confidence', 0.0)
                
                # Check for common OCR issues
                char_issues = self._check_character_confusion(text)
                if char_issues:
                    issues.append(f"Character confusion: {char_issues}")
                    confidence = max(0.3, avg_conf - 0.2)
                else:
                    confidence = avg_conf
                
                # Check text length
                if len(text) < 50:
                    issues.append("Very little text extracted")
                    confidence = max(0.2, confidence - 0.1)
                
                # Check for gibberish (repeating characters)
                if self._check_gibberish(text):
                    issues.append("Possible gibberish detected")
                    confidence = max(0.1, confidence - 0.3)
                    
            elif isinstance(ocr_result, str):
                text = ocr_result
                # Simple text-based assessment
                if len(text) > 100:
                    confidence = 0.6
                elif len(text) > 20:
                    confidence = 0.4
                else:
                    confidence = 0.2
                    issues.append("Minimal text extracted")
        
        except Exception as e:
            logger.warning(f"Page OCR assessment failed for page {page_num}: {e}")
            confidence = 0.3
            issues.append(f"Assessment error: {str(e)}")
        
        return confidence, issues
    
    def _check_character_confusion(self, text: str) -> List[str]:
        """Check for commonly confused characters"""
        issues = []
        
        for i, char in enumerate(text):
            if char in self.confusing_characters:
                # Check if surrounding context suggests confusion
                if i > 0 and i < len(text) - 1:
                    context = text[i-1:i+2]
                    if any(confused in context for confused in self.confusing_characters[char]):
                        issues.append(f"Position {i}: '{char}' could be confused")
        
        return issues[:5]  # Limit to top 5 issues
    
    def _check_gibberish(self, text: str) -> bool:
        """Check for gibberish text"""
        # Check for excessive repeating characters
        for i in range(len(text) - 3):
            if text[i] == text[i+1] == text[i+2] == text[i+3]:
                return True
        
        # Check for non-alphanumeric sequences
        non_alpha_seq = re.findall(r'[^a-zA-Z0-9\s\.\,]{4,}', text)
        return len(non_alpha_seq) > 2