import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import os
from PIL import Image
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ImageProcessor:
    """Image processing service"""
    
    def __init__(self):
        self.max_image_size = (2000, 2000)  # Maximum dimensions
    
    async def process_image(self, image_path: str) -> Optional[List[np.ndarray]]:
        """
        Process single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            List containing single image as numpy array
        """
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Check file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Load image
            image = self._load_image(image_path)
            
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Preprocess image
            processed = self._preprocess_image(image)
            
            return [processed]
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return None
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file"""
        try:
            # Try OpenCV first
            image = cv2.imread(image_path)
            
            if image is None:
                # Fallback to PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                
                # Convert RGB to BGR for OpenCV compatibility
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            logger.error(f"Image loading failed: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for OCR and CV"""
        try:
            # Resize if too large
            height, width = image.shape[:2]
            if height > self.max_image_size[0] or width > self.max_image_size[1]:
                scale = min(
                    self.max_image_size[0] / height,
                    self.max_image_size[1] / width
                )
                new_size = (int(width * scale), int(height * scale))
                image = cv2.resize(image, new_size)
            
            # Enhance contrast for better OCR
            if len(image.shape) == 3:
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Convert back to BGR if needed
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {e}")
            return image
    
    async def process_multiple_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Process multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of processed images
        """
        try:
            logger.info(f"Processing {len(image_paths)} images")
            
            images = []
            for path in image_paths:
                processed = await self.process_image(path)
                if processed:
                    images.extend(processed)
            
            logger.info(f"Processed {len(images)} images successfully")
            return images
            
        except Exception as e:
            logger.error(f"Multiple image processing failed: {e}")
            return []
    
    async def extract_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Extract metadata from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image metadata
        """
        try:
            metadata = {
                "filename": os.path.basename(image_path),
                "file_size": os.path.getsize(image_path),
                "file_extension": os.path.splitext(image_path)[1].lower()
            }
            
            # Load image to get dimensions
            image = self._load_image(image_path)
            if image is not None:
                metadata.update({
                    "dimensions": {
                        "height": image.shape[0],
                        "width": image.shape[1],
                        "channels": image.shape[2] if len(image.shape) == 3 else 1
                    },
                    "dtype": str(image.dtype)
                })
            
            # Try to get EXIF data
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.open(image_path)
                exif_data = pil_img._getexif()
                if exif_data:
                    metadata["exif"] = {
                        str(k): exif_data[k] for k in exif_data if isinstance(k, int)
                    }
            except:
                pass  # EXIF data not available
            
            return metadata
            
        except Exception as e:
            logger.error(f"Image metadata extraction failed: {e}")
            return {"error": str(e)}
    
    async def detect_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect image quality metrics
        
        Args:
            image: Input image
            
        Returns:
            Quality metrics
        """
        try:
            metrics = {}
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 1. Sharpness (variance of Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            metrics["sharpness"] = float(laplacian.var())
            
            # 2. Brightness
            metrics["brightness"] = float(np.mean(gray))
            
            # 3. Contrast (standard deviation)
            metrics["contrast"] = float(np.std(gray))
            
            # 4. Noise (estimate)
            # Apply blur and compare
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
            metrics["noise_level"] = float(noise)
            
            # 5. Blur detection
            # Using Brenner's focus measure
            brenner = np.sum(np.square(np.diff(gray, axis=1)))
            metrics["focus_measure"] = float(brenner / (gray.shape[0] * gray.shape[1]))
            
            # Overall quality score (weighted average)
            weights = {
                "sharpness": 0.3,
                "contrast": 0.3,
                "noise_level": 0.2,
                "focus_measure": 0.2
            }
            
            # Normalize metrics
            normalized = {}
            for key in weights.keys():
                if key in metrics:
                    # Simple normalization (assuming reasonable ranges)
                    if key == "sharpness":
                        normalized[key] = min(metrics[key] / 1000, 1.0)
                    elif key == "noise_level":
                        normalized[key] = max(0, 1 - min(metrics[key] / 50, 1.0))
                    elif key == "focus_measure":
                        normalized[key] = min(metrics[key] / 1000, 1.0)
                    else:
                        normalized[key] = min(metrics[key] / 255, 1.0)
            
            # Calculate weighted score
            quality_score = sum(
                normalized.get(key, 0.5) * weight 
                for key, weight in weights.items()
            )
            
            metrics["quality_score"] = float(quality_score)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Image quality detection failed: {e}")
            return {"error": str(e)}