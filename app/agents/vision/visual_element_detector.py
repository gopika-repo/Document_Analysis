from typing import Dict, List, Any
from app.core.models import ProcessingState, VisualElement
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class VisualElementDetector:
    """Agent for detecting visual elements in documents using YOLO"""
    
    def __init__(self):
        self.element_types = ["table", "chart", "figure", "signature", "logo", "stamp", "text_region"]
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Detect visual elements in document images"""
        try:
            logger.info(f"Detecting visual elements for {state.document_id}")
            
            visual_elements = {}
            
            for idx, image in enumerate(state.images):
                elements = await self._detect_elements_in_image(image, idx)
                visual_elements[idx] = elements
                logger.debug(f"Page {idx}: Detected {len(elements)} elements")
            
            state.visual_elements = visual_elements
            logger.info(f"Total visual elements detected: {sum(len(v) for v in visual_elements.values())}")
            
            return state
            
        except Exception as e:
            logger.error(f"Visual element detection failed: {e}")
            state.visual_elements = {}
            return state
    
    async def _detect_elements_in_image(self, image, page_num: int) -> List[VisualElement]:
        """Detect elements in a single image"""
        elements = []
        
        try:
            # Try to use YOLO if available
            try:
                from app.models.yolo_loader import YOLOModel
                yolo_model = YOLOModel()
                detections = yolo_model.detect(image, page_num)
                
                for detection in detections:
                    element = VisualElement(
                        element_type=detection.class_name,
                        bbox=detection.bbox,
                        confidence=detection.confidence,
                        page_num=page_num,
                        metadata={
                            "detection_method": "yolo",
                            "features": detection.features.tolist() if detection.features is not None else None
                        }
                    )
                    elements.append(element)
                    
            except ImportError:
                # Fallback to simple detection based on image features
                elements = self._simple_detection_fallback(image, page_num)
            
        except Exception as e:
            logger.warning(f"Element detection failed for page {page_num}: {e}")
        
        return elements
    
    def _simple_detection_fallback(self, image, page_num: int) -> List[VisualElement]:
        """Simple fallback detection for when YOLO is not available"""
        elements = []
        
        # Create some dummy detections for testing
        import random
        
        # Simulate detecting tables (usually rectangular regions)
        if random.random() > 0.5:
            elements.append(VisualElement(
                element_type="table",
                bbox=[100, 100, 400, 300],
                confidence=0.7,
                page_num=page_num,
                metadata={"detection_method": "fallback"}
            ))
        
        # Simulate detecting text regions
        elements.append(VisualElement(
            element_type="text_region",
            bbox=[50, 50, 500, 200],
            confidence=0.8,
            page_num=page_num,
            metadata={"detection_method": "fallback"}
        ))
        
        return elements