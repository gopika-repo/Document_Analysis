import logging
import sys
from pythonjsonlogger import jsonlogger
from datetime import datetime
import os

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup JSON logger for structured logging
    
    Args:
        name: Logger name
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s %(filename)s %(lineno)d',
        rename_fields={
            'asctime': 'timestamp',
            'name': 'logger',
            'levelname': 'level',
            'filename': 'file',
            'lineno': 'line'
        }
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler for errors
    try:
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler(
            f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to setup file logging: {e}")
    
    return logger

class APILogger:
    """Logger specifically for API endpoints"""
    
    def __init__(self, name: str = "api"):
        self.logger = setup_logger(name)
    
    def log_request(self, 
                   method: str, 
                   path: str, 
                   client_ip: str, 
                   user_agent: str = None):
        """Log API request"""
        self.logger.info(
            "API request",
            extra={
                "method": method,
                "path": path,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "type": "request"
            }
        )
    
    def log_response(self, 
                    method: str, 
                    path: str, 
                    status_code: int, 
                    duration_ms: float):
        """Log API response"""
        level = logging.INFO if status_code < 400 else logging.WARNING
        self.logger.log(
            level,
            "API response",
            extra={
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "type": "response"
            }
        )
    
    def log_error(self, 
                 method: str, 
                 path: str, 
                 error: str, 
                 traceback: str = None):
        """Log API error"""
        self.logger.error(
            "API error",
            extra={
                "method": method,
                "path": path,
                "error": error,
                "traceback": traceback,
                "type": "error"
            }
        )