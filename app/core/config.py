import os
from typing import Optional, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Vision-Fusion Document Intelligence"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # File Processing
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    TEMP_DIR: str = "temp"
    
    # OCR Settings
    TESSERACT_PATH: Optional[str] = None
    OCR_DPI: int = 300
    OCR_CONFIDENCE_THRESHOLD: float = 0.7
    
    # YOLO Settings
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_CLASSES: List[str] = [
        "table", "chart", "figure", "signature", 
        "logo", "stamp", "text_region", "header", "footer"
    ]
    
    # Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "document_embeddings"
    
    # Embeddings
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    IMAGE_EMBEDDING_MODEL: str = "clip-ViT-B-32"
    EMBEDDING_DIMENSION: int = 384
    
    # LLM Settings (Optional)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    
    # Agent Settings
    AGENT_CONFIDENCE_THRESHOLD: float = 0.6
    CONTRADICTION_SEVERITY_THRESHOLD: float = 0.7
    RISK_SCORE_THRESHOLD: float = 0.8
    
    # Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_TTL: int = 3600
    
    # Monitoring
    LOG_LEVEL: str = "INFO"
    METRICS_ENABLED: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()