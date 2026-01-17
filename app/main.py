from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime

# Import from app modules with try-except
try:
    from app.api.routes import router
    from app.core.config import settings
    from app.utils.logger import setup_logger
    HAS_MODULES = True
except ImportError as e:
    print(f"Warning: Some modules not found: {e}")
    HAS_MODULES = False
    # Create minimal versions
    class Settings:
        APP_NAME = "Vision-Fusion Document Intelligence"
        VERSION = "1.0.0"
        DEBUG = True
        ENVIRONMENT = "development"
        HOST = "0.0.0.0"
        PORT = 8000
        WORKERS = 1
    
    settings = Settings()
    
    # Create minimal router
    from fastapi import APIRouter
    router = APIRouter()

logger = None
if HAS_MODULES:
    logger = setup_logger(__name__)
else:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Initialize services if available
    try:
        from app.services.document_processor import DocumentProcessor
        app.state.document_processor = DocumentProcessor()
        logger.info("DocumentProcessor service initialized")
    except ImportError:
        logger.warning("DocumentProcessor not available, using stub")
        app.state.document_processor = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Multi-Modal Document Intelligence System with AI Document Auditing",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers if available
if HAS_MODULES:
    app.include_router(router, prefix="/api/v1")

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "upload": "/api/v1/upload",
            "process": "/api/v1/process",
            "status": "/api/v1/status/{job_id}",
            "results": "/api/v1/results/{document_id}",
            "search": "/api/v1/search",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "timestamp": datetime.now().isoformat()
    }

# Basic API endpoints if router not available
if not HAS_MODULES:
    @app.post("/api/v1/upload")
    async def upload_document():
        return {
            "success": False,
            "message": "API modules not fully loaded",
            "document_id": "stub"
        }
    
    @app.get("/api/v1/status/{job_id}")
    async def get_status(job_id: str):
        return {
            "document_id": job_id,
            "status": "unknown",
            "message": "Service initializing"
        }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )