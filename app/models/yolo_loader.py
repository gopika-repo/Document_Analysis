import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import torch
from dataclasses import dataclass
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class DetectionResult:
    """Data class for detection results"""
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    page_num: int
    features: Optional[np.ndarray] = None

class YOLOModel:
    """YOLOv8 Model Wrapper for Document Element Detection"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.confidence_threshold = settings.YOLO_CONFIDENCE_THRESHOLD
        self.classes = settings.DETECTION_CLASSES
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model()
        
    def _load_model(self) -> YOLO:
        """Load YOLOv8 model"""
        try:
            logger.info(f"Loading YOLO model from {self.model_path} on {self.device}")
            model = YOLO(self.model_path)
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, image: np.ndarray, page_num: int = 0) -> List[DetectionResult]:
        """
        Detect document elements in an image
        
        Args:
            image: Input image as numpy array
            page_num: Page number for reference
            
        Returns:
            List of detection results
        """
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        class_name = self.model.names[class_id]
                        
                        # Convert box to list
                        bbox = [int(b) for b in box]
                        
                        # Extract region features
                        region = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        features = self._extract_features(region)
                        
                        detection = DetectionResult(
                            class_name=class_name,
                            confidence=float(conf),
                            bbox=bbox,
                            page_num=page_num,
                            features=features
                        )
                        detections.append(detection)
            
            logger.info(f"Detected {len(detections)} elements on page {page_num}")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _extract_features(self, region: np.ndarray) -> Optional[np.ndarray]:
        """Extract visual features from region"""
        try:
            # Resize for consistent feature extraction
            region_resized = cv2.resize(region, (224, 224))
            
            # Simple feature extraction
            # Convert to grayscale
            gray = cv2.cvtColor(region_resized, cv2.COLOR_BGR2GRAY)
            
            # Compute histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Try to use scikit-image for LBP, fallback if not available
            try:
                from skimage.feature import local_binary_pattern
                lbp = local_binary_pattern(gray, 8, 1, method='uniform')
                lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
                lbp_hist = lbp_hist / lbp_hist.sum()
                # Combine features
                features = np.concatenate([hist, lbp_hist])
            except ImportError:
                # Fallback: just use histogram features
                logger.debug("scikit-image not available, using histogram features only")
                features = hist
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None
    
    def batch_detect(self, images: List[np.ndarray]) -> Dict[int, List[DetectionResult]]:
        """
        Batch detection for multiple pages
        
        Args:
            images: List of images
            
        Returns:
            Dictionary mapping page numbers to detections
        """
        results = {}
        for idx, image in enumerate(images):
            detections = self.detect(image, page_num=idx)
            results[idx] = detections
        return results