"""
Core module for document processing system.
Contains models, interfaces, and base classes.
"""

from app.core.models import (
    # Base classes
    ProcessingState,
    Document,
    Image,
    
    # Models
    QualityScore,
    VisualElement,
    TableStructure,
    SignatureVerification,
    ExtractedEntities,
    SemanticAnalysis,
    AlignedData,
    FieldConfidences,
    TemporalConsistency,
    Contradiction,
    SystemFeedback,
    ReviewItem,
    Explanation,
    
    # Enums
    ProcessingStage,
    ContradictionType,
    SeverityLevel,
    FeedbackType,
    ReviewPriority,
    ElementType
)

from app.core.interfaces import (
    # Interfaces
    ProcessingAgent,
    ValidationRule,
    LearningComponent,
    
    # Base agents
    BaseAgent,
    BaseValidationAgent,
    BaseLearningAgent
)

from app.core.exceptions import (
    # Exceptions
    ProcessingError,
    ValidationError,
    LearningError,
    ConfigurationError,
    ResourceError
)

from app.core.config import (
    # Configuration
    ProcessingConfig,
    AgentConfig,
    ValidationConfig,
    LearningConfig,
    get_config
)

__all__ = [
    # Models
    'ProcessingState',
    'Document',
    'Image',
    'QualityScore',
    'VisualElement',
    'TableStructure',
    'SignatureVerification',
    'ExtractedEntities',
    'SemanticAnalysis',
    'AlignedData',
    'FieldConfidences',
    'TemporalConsistency',
    'Contradiction',
    'SystemFeedback',
    'ReviewItem',
    'Explanation',
    
    # Enums
    'ProcessingStage',
    'ContradictionType',
    'SeverityLevel',
    'FeedbackType',
    'ReviewPriority',
    'ElementType',
    
    # Interfaces
    'ProcessingAgent',
    'ValidationRule',
    'LearningComponent',
    'BaseAgent',
    'BaseValidationAgent',
    'BaseLearningAgent',
    
    # Exceptions
    'ProcessingError',
    'ValidationError',
    'LearningError',
    'ConfigurationError',
    'ResourceError',
    
    # Configuration
    'ProcessingConfig',
    'AgentConfig',
    'ValidationConfig',
    'LearningConfig',
    'get_config'
]