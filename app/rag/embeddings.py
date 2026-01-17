import cv2
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Union
import torch
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingEngine:
    """Embedding engine for multi-modal embeddings"""
    
    def __init__(self):
        self.text_model = self._load_text_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _load_text_model(self) -> SentenceTransformer:
        """Load text embedding model"""
        try:
            logger.info(f"Loading text embedding model: {settings.TEXT_EMBEDDING_MODEL}")
            model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL)
            return model
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
            raise
    
    def generate_text_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate text embeddings
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = self.text_model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Text embedding generation failed: {e}")
            raise
    
    def generate_visual_embeddings(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Generate visual embeddings from images
        
        Args:
            images: List of numpy arrays representing images
            
        Returns:
            Numpy array of embeddings
        """
        try:
            # Simple visual embedding using color histograms
            # In production, use a proper vision model like CLIP
            
            embeddings = []
            for img in images:
                # Convert to RGB if needed
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                
                # Resize for consistency
                img_resized = cv2.resize(img, (224, 224))
                
                # Extract features (simplified - use CLIP in production)
                # 1. Color histogram
                hist_r = cv2.calcHist([img_resized], [0], None, [64], [0, 256])
                hist_g = cv2.calcHist([img_resized], [1], None, [64], [0, 256])
                hist_b = cv2.calcHist([img_resized], [2], None, [64], [0, 256])
                
                # Normalize
                hist_r = cv2.normalize(hist_r, hist_r).flatten()
                hist_g = cv2.normalize(hist_g, hist_g).flatten()
                hist_b = cv2.normalize(hist_b, hist_b).flatten()
                
                # Combine
                features = np.concatenate([hist_r, hist_g, hist_b])
                embeddings.append(features)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Visual embedding generation failed: {e}")
            raise
    
    def generate_multi_modal_embeddings(self, 
                                       text: str, 
                                       image: np.ndarray = None) -> Dict[str, Any]:
        """
        Generate multi-modal embeddings
        
        Args:
            text: Text content
            image: Optional image content
            
        Returns:
            Dictionary with text and visual embeddings
        """
        try:
            result = {
                "text_embedding": self.generate_text_embeddings(text)[0].tolist()
            }
            
            if image is not None:
                result["visual_embedding"] = self.generate_visual_embeddings([image])[0].tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-modal embedding generation failed: {e}")
            raise