import os
from typing import Optional
import json

# Try to import pydantic_settings, fallback to regular pydantic
try:
    from pydantic_settings import BaseSettings
    HAS_PYDANTIC_SETTINGS = True
except ImportError:
    from pydantic import BaseModel as BaseSettings
    HAS_PYDANTIC_SETTINGS = False
    print("Warning: pydantic-settings not installed, using basic config")

class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "Vision-Fusion Document Intelligence"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # File Upload Settings
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}
    
    # OCR Settings
    TESSERACT_PATH: Optional[str] = os.getenv("TESSERACT_PATH")
    OCR_CONFIDENCE_THRESHOLD: float = 0.85
    OCR_DPI: int = 300
    
    # YOLO Model Settings
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    DETECTION_CLASSES: list = [
        "table", "chart", "diagram", "signature", 
        "handwritten", "logo", "stamp", "text_region"
    ]
    
    # Qdrant Settings
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = "document_embeddings"
    
    # LLM Settings
    GROK_API_KEY: Optional[str] = os.getenv("GROK_API_KEY")
    GROK_API_URL: str = "https://api.x.ai/v1/chat/completions"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    
    # Embedding Settings
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_SIZE: int = 384
    
    # Redis Cache
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_TTL: int = 3600  # 1 hour
    
    # Load from .env file if pydantic-settings is available
    if HAS_PYDANTIC_SETTINGS:
        class Config:
            env_file = ".env"
    else:
        # Manual loading from .env
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value
            except FileNotFoundError:
                pass  # .env file doesn't exist

settings = Settings()