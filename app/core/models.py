from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import uuid

class DocumentType(str, Enum):
    FINANCIAL_REPORT = "financial_report"
    INVOICE = "invoice"
    CONTRACT = "contract"
    FORM = "form"
    RESEARCH_PAPER = "research_paper"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class QualityScore(BaseModel):
    sharpness: float = Field(..., ge=0, le=1)
    brightness: float = Field(..., ge=0, le=1)
    contrast: float = Field(..., ge=0, le=1)
    noise_level: float = Field(..., ge=0, le=1)
    overall: float = Field(..., ge=0, le=1)

class ContradictionType(str, Enum):
    CHART_TEXT_CONFLICT = "chart_text_conflict"
    NUMERIC_INCONSISTENCY = "numeric_inconsistency"
    DATE_MISMATCH = "date_mismatch"
    SIGNATURE_ABSENCE = "signature_absence"
    CALCULATION_ERROR = "calculation_error"
    SUMMARY_TABLE_MISMATCH = "summary_table_mismatch"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExtractedField(BaseModel):
    value: Any
    confidence: float = Field(..., ge=0, le=1)
    sources: List[str]
    modalities: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VisualElement(BaseModel):
    element_type: str
    bbox: List[int]
    confidence: float
    page_num: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Contradiction(BaseModel):
    contradiction_type: ContradictionType
    severity: SeverityLevel
    field_a: str
    field_b: str
    value_a: Any
    value_b: Any
    explanation: str
    confidence: float

class ProcessingState(BaseModel):
    """State for LangGraph workflow"""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    images: List[Any] = Field(default_factory=list)
    
    # Preprocessing results
    quality_scores: Dict[int, QualityScore] = Field(default_factory=dict)
    document_type: Optional[DocumentType] = None
    layout_strategy: Optional[str] = None
    
    # Vision results
    visual_elements: Dict[int, List[VisualElement]] = Field(default_factory=dict)
    chart_analysis: Dict[str, Any] = Field(default_factory=dict)
    table_structures: Dict[str, Any] = Field(default_factory=dict)
    signature_verification: Dict[str, Any] = Field(default_factory=dict)
    
    # Text results
    ocr_results: Dict[int, Any] = Field(default_factory=dict)
    ocr_confidence: Dict[int, float] = Field(default_factory=dict)
    extracted_entities: Dict[str, List[str]] = Field(default_factory=dict)
    semantic_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    # Fusion results
    aligned_data: Dict[str, Any] = Field(default_factory=dict)
    field_confidences: Dict[str, float] = Field(default_factory=dict)
    temporal_consistency: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation results
    contradictions: List[Contradiction] = Field(default_factory=list)
    risk_score: float = Field(default=0.0, ge=0, le=1)
    compliance_issues: List[str] = Field(default_factory=list)
    
    # Explainability results
    explanations: Dict[str, str] = Field(default_factory=dict)
    review_recommendations: List[str] = Field(default_factory=list)
    
    # Final output
    extracted_fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    processing_start: datetime = Field(default_factory=datetime.now)
    processing_end: Optional[datetime] = None