from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from .models import DocumentType, Contradiction, SeverityLevel

class DocumentUploadRequest(BaseModel):
    file_name: str
    file_size: int
    file_type: str

class DocumentUploadResponse(BaseModel):
    document_id: str
    message: str
    status_endpoint: str
    estimated_processing_time: int = 60

class ProcessingStatusResponse(BaseModel):
    document_id: str
    status: str
    progress: float = Field(0.0, ge=0, le=1)
    current_step: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    errors: List[str] = Field(default_factory=list)

class ProcessingResultsResponse(BaseModel):
    success: bool
    document_id: str
    document_type: DocumentType
    processing_time: float
    
    # Extracted data
    extracted_fields: Dict[str, Any]
    confidence_scores: Dict[str, float]
    
    # Validation results
    contradictions: List[Contradiction]
    risk_score: float = Field(0.0, ge=0, le=1)
    integrity_score: float = Field(0.0, ge=0, le=1)
    
    # Recommendations
    recommendations: List[str]
    required_actions: List[str]
    
    # Metadata
    processing_metadata: Dict[str, Any]
    agent_contributions: Dict[str, float]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class RAGSearchRequest(BaseModel):
    query: str
    query_type: str = "hybrid"
    limit: int = Field(10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None

class RAGSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    explanation: str
    confidence: float = Field(..., ge=0, le=1)
    sources_used: List[str]