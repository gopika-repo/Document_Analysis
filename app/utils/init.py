"""
Utility functions and helpers for the document processing system.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
import hashlib

# Configure logging
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up and configure a logger"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger

# File utilities
def save_json(data: Any, filepath: Union[str, Path]) -> bool:
    """Save data as JSON file"""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON to {filepath}: {e}")
        return False

def load_json(filepath: Union[str, Path]) -> Optional[Any]:
    """Load data from JSON file"""
    try:
        path = Path(filepath)
        if not path.exists():
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON from {filepath}: {e}")
        return None

# Data validation
def validate_required_fields(data: Dict[str, Any], 
                           required_fields: List[str]) -> List[str]:
    """Validate that required fields are present in data"""
    missing_fields = []
    
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    return missing_fields

def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize dictionary by removing None values and empty strings"""
    sanitized = {}
    
    for key, value in data.items():
        if value is not None:
            if isinstance(value, str) and value.strip() == "":
                continue
            sanitized[key] = value
    
    return sanitized

# String utilities
def truncate_string(text: str, max_length: int = 100) -> str:
    """Truncate string to maximum length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def generate_hash(data: Union[str, bytes]) -> str:
    """Generate SHA256 hash for data"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()

# Date/time utilities
def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """Format timestamp as ISO string"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.isoformat()

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime"""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None

# Performance utilities
class Timer:
    """Context manager for timing code execution"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        logging.debug(f"{self.name} completed in {elapsed:.3f} seconds")
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

# Configuration utilities
def get_env_variable(name: str, default: Any = None) -> Any:
    """Get environment variable with default"""
    import os
    return os.environ.get(name, default)

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file"""
    config = load_json(config_path)
    if config is None:
        config = {}
    
    # Override with environment variables
    import os
    for key in config.keys():
        env_key = key.upper()
        if env_key in os.environ:
            config[key] = os.environ[env_key]
    
    return config

# Serialization utilities
class EnhancedJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder that handles more data types"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        
        return super().default(obj)

def serialize_to_json(data: Any) -> str:
    """Serialize data to JSON string"""
    return json.dumps(data, cls=EnhancedJSONEncoder, indent=2)

def deserialize_from_json(json_str: str) -> Any:
    """Deserialize data from JSON string"""
    return json.loads(json_str)

# Math utilities
def calculate_percentage(part: float, whole: float) -> float:
    """Calculate percentage"""
    if whole == 0:
        return 0.0
    return (part / whole) * 100

def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-1 range"""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))

# File I/O utilities
def read_file(filepath: Union[str, Path]) -> Optional[str]:
    """Read text file"""
    try:
        path = Path(filepath)
        return path.read_text(encoding='utf-8')
    except Exception as e:
        logging.error(f"Failed to read file {filepath}: {e}")
        return None

def write_file(filepath: Union[str, Path], content: str) -> bool:
    """Write text file"""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return True
    except Exception as e:
        logging.error(f"Failed to write file {filepath}: {e}")
        return False

__all__ = [
    'setup_logger',
    'save_json',
    'load_json',
    'validate_required_fields',
    'sanitize_dict',
    'truncate_string',
    'generate_hash',
    'format_timestamp',
    'parse_timestamp',
    'Timer',
    'get_env_variable',
    'load_config',
    'EnhancedJSONEncoder',
    'serialize_to_json',
    'deserialize_from_json',
    'calculate_percentage',
    'normalize_value',
    'clamp',
    'read_file',
    'write_file'
]