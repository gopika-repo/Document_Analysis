import cv2
import numpy as np
from typing import Dict, List, Any
from langgraph.graph import END
from pydantic import BaseModel, Field
from app.core.models import ProcessingState, QualityScore
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentQualityAgent:
    """Agent for assessing document quality before processing"""
    
    def __init__(self):
        self.quality_threshold = 0.5
    
    async def assess_quality(self, state: ProcessingState) -> ProcessingState:
        """Assess quality of document images"""
        try:
            logger.info(f"Assessing document quality for {state.document_id}")
            
            quality_scores = {}
            for idx, image in enumerate(state.images):
                if isinstance(image, np.ndarray):
                    score = self._calculate_image_quality(image)
                    quality_scores[idx] = score
                    logger.debug(f"Page {idx} quality: {score.overall:.2f}")
            
            state.quality_scores = quality_scores
            
            # Check if any page fails quality threshold
            low_quality_pages = [
                idx for idx, score in quality_scores.items() 
                if score.overall < self.quality_threshold
            ]
            
            if low_quality_pages:
                state.errors.append(
                    f"Low quality pages detected: {low_quality_pages}. "
                    f"May affect extraction accuracy."
                )
            
            logger.info(f"Quality assessment completed for {len(quality_scores)} pages")
            return state
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            state.errors.append(f"Quality assessment error: {str(e)}")
            return state
    
    def _calculate_image_quality(self, image: np.ndarray) -> QualityScore:
        """Calculate various quality metrics for an image"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 1. Sharpness (variance of Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = min(laplacian.var() / 1000, 1.0)
            
            # 2. Brightness
            brightness = np.mean(gray) / 255.0
            
            # 3. Contrast (normalized standard deviation)
            contrast = np.std(gray) / 128.0
            
            # 4. Noise level (difference from smoothed image)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = np.mean(np.abs(gray.astype(float) - blurred.astype(float))) / 128.0
            noise_level = max(0, 1 - noise)
            
            # 5. Focus measure (Brenner's)
            brenner = np.sum(np.square(np.diff(gray, axis=1)))
            focus = min(brenner / (gray.shape[0] * gray.shape[1] * 1000), 1.0)
            
            # Overall weighted score
            weights = {
                "sharpness": 0.25,
                "focus": 0.25,
                "contrast": 0.20,
                "brightness": 0.15,
                "noise_level": 0.15
            }
            
            overall = (
                sharpness * weights["sharpness"] +
                focus * weights["focus"] +
                contrast * weights["contrast"] +
                brightness * weights["brightness"] +
                noise_level * weights["noise_level"]
            )
            
            return QualityScore(
                sharpness=float(sharpness),
                brightness=float(brightness),
                contrast=float(contrast),
                noise_level=float(noise_level),
                overall=float(overall)
            )
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return QualityScore(
                sharpness=0.5,
                brightness=0.5,
                contrast=0.5,
                noise_level=0.5,
                overall=0.5
            )