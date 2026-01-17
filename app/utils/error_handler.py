from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import traceback
from typing import Union
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AppException(Exception):
    """Custom application exception"""
    def __init__(self, 
                 message: str, 
                 status_code: int = 500, 
                 error_code: str = None,
                 details: dict = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class OCRException(AppException):
    """OCR processing exception"""
    pass

class CVException(AppException):
    """Computer Vision exception"""
    pass

class AgentException(AppException):
    """Agent processing exception"""
    pass

class ValidationException(AppException):
    """Validation exception"""
    pass

def setup_exception_handlers(app: FastAPI):
    """Setup global exception handlers"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.error(
            f"HTTP Exception: {exc.detail}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code,
                "detail": exc.detail
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": request.url.path
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation exceptions"""
        errors = []
        for error in exc.errors():
            errors.append({
                "loc": error["loc"],
                "msg": error["msg"],
                "type": error["type"]
            })
        
        logger.warning(
            f"Validation error: {errors}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "errors": errors
            }
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Validation error",
                "errors": errors,
                "path": request.url.path
            }
        )
    
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        """Handle custom application exceptions"""
        logger.error(
            f"Application Exception: {exc.message}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code,
                "error_code": exc.error_code,
                "details": exc.details
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.message,
                "error_code": exc.error_code,
                "details": exc.details,
                "status_code": exc.status_code,
                "path": request.url.path
            }
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions"""
        error_traceback = traceback.format_exc()
        
        logger.error(
            f"Unhandled Exception: {str(exc)}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "error": str(exc),
                "traceback": error_traceback
            }
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "status_code": 500,
                "path": request.url.path,
                "detail": str(exc) if app.debug else "An internal error occurred"
            }
        )

def handle_api_error(func):
    """Decorator to handle API errors"""
    from functools import wraps
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except AppException as e:
            raise e
        except Exception as e:
            logger.error(f"API error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return wrapper