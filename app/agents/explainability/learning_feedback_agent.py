from typing import Dict, List, Any, Optional
from datetime import datetime
from app.core.models import ProcessingState, FeedbackType, SystemFeedback
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class LearningFeedbackAgent:
    """Agent for collecting and processing system learning feedback"""
    
    def __init__(self):
        self.feedback_categories = {
            "confidence_calibration": "Adjust confidence scoring",
            "entity_extraction": "Improve entity recognition",
            "contradiction_detection": "Enhance contradiction detection",
            "document_classification": "Better document type identification",
            "quality_assessment": "Refine quality metrics",
            "risk_assessment": "Improve risk evaluation"
        }
        
        # In production, this would connect to a database
        self.feedback_history = []
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Process feedback and update learning"""
        try:
            logger.info(f"Processing learning feedback for {state.document_id}")
            
            # Generate feedback based on processing results
            system_feedback = self._generate_feedback(state)
            
            # Analyze feedback for system improvements
            improvements = self._analyze_feedback_for_improvements(system_feedback)
            
            # Update processing metadata with feedback
            state.processing_metadata = state.processing_metadata or {}
            state.processing_metadata["system_feedback"] = system_feedback
            state.processing_metadata["suggested_improvements"] = improvements
            
            # Store feedback for learning (in production, this would be saved to DB)
            self._store_feedback(state.document_id, system_feedback)
            
            logger.info(f"Generated {len(system_feedback)} feedback items")
            
            return state
            
        except Exception as e:
            logger.error(f"Learning feedback processing failed: {e}")
            return state
    
    def _generate_feedback(self, state: ProcessingState) -> List[SystemFeedback]:
        """Generate system feedback based on processing results"""
        feedback_items = []
        
        # Feedback on confidence calibration
        confidence_feedback = self._generate_confidence_feedback(state)
        if confidence_feedback:
            feedback_items.append(confidence_feedback)
        
        # Feedback on entity extraction
        entity_feedback = self._generate_entity_feedback(state)
        if entity_feedback:
            feedback_items.append(entity_feedback)
        
        # Feedback on contradiction detection
        contradiction_feedback = self._generate_contradiction_feedback(state)
        if contradiction_feedback:
            feedback_items.append(contradiction_feedback)
        
        # Feedback on document classification
        classification_feedback = self._generate_classification_feedback(state)
        if classification_feedback:
            feedback_items.append(classification_feedback)
        
        # Feedback on quality assessment
        quality_feedback = self._generate_quality_feedback(state)
        if quality_feedback:
            feedback_items.append(quality_feedback)
        
        # Feedback on risk assessment
        risk_feedback = self._generate_risk_feedback(state)
        if risk_feedback:
            feedback_items.append(risk_feedback)
        
        return feedback_items
    
    def _generate_confidence_feedback(self, state: ProcessingState) -> Optional[SystemFeedback]:
        """Generate feedback on confidence calibration"""
        try:
            if hasattr(state, 'field_confidences') and state.field_confidences:
                overall_confidence = state.field_confidences.get('overall', 0.5)
                
                # Check if confidence seems appropriate
                confidence_indicators = []
                
                # Look at OCR confidence
                if hasattr(state, 'ocr_confidence') and state.ocr_confidence:
                    avg_ocr = sum(state.ocr_confidence.values()) / len(state.ocr_confidence)
                    if overall_confidence > avg_ocr * 1.5:
                        confidence_indicators.append(f"Overall confidence {overall_confidence:.0%} seems high compared to OCR confidence {avg_ocr:.0%}")
                
                # Look at contradiction count
                if hasattr(state, 'contradictions') and state.contradictions:
                    if overall_confidence > 0.7 and len(state.contradictions) > 2:
                        confidence_indicators.append(f"High confidence ({overall_confidence:.0%}) with {len(state.contradictions)} contradictions")
                
                if confidence_indicators:
                    return SystemFeedback(
                        feedback_type=FeedbackType.CONFIDENCE_CALIBRATION,
                        category="confidence_calibration",
                        description="Confidence calibration may need adjustment",
                        details="; ".join(confidence_indicators),
                        severity="medium",
                        suggested_action="Review confidence weighting algorithms",
                        confidence=0.6
                    )
        except Exception as e:
            logger.warning(f"Confidence feedback generation failed: {e}")
        
        return None
    
    def _generate_entity_feedback(self, state: ProcessingState) -> Optional[SystemFeedback]:
        """Generate feedback on entity extraction"""
        try:
            if hasattr(state, 'extracted_entities') and state.extracted_entities:
                entity_stats = {}
                for entity_type, entities in state.extracted_entities.items():
                    entity_stats[entity_type] = len(entities)
                
                # Check for unusual patterns
                feedback_items = []
                
                # Many dates but no years mentioned
                if entity_stats.get('dates', 0) > 5:
                    # Check if dates include years
                    dates = state.extracted_entities.get('dates', [])
                    year_dates = [d for d in dates if any(str(year) in d for year in range(1900, 2100))]
                    if len(year_dates) < len(dates) / 2:
                        feedback_items.append("Many dates extracted but few include years")
                
                # Many amounts but no currency
                if entity_stats.get('amounts', 0) > 3:
                    amounts = state.extracted_entities.get('amounts', [])
                    currency_amounts = [a for a in amounts if any(c in a for c in ['$', '₹', '€', '£', 'USD', 'EUR'])]
                    if len(currency_amounts) < len(amounts) / 2:
                        feedback_items.append("Many amounts extracted but few specify currency")
                
                if feedback_items:
                    return SystemFeedback(
                        feedback_type=FeedbackType.ENTITY_EXTRACTION,
                        category="entity_extraction",
                        description="Entity extraction patterns suggest improvements",
                        details="; ".join(feedback_items),
                        severity="low",
                        suggested_action="Improve entity normalization and context detection",
                        confidence=0.5
                    )
        except Exception as e:
            logger.warning(f"Entity feedback generation failed: {e}")
        
        return None
    
    def _generate_contradiction_feedback(self, state: ProcessingState) -> Optional[SystemFeedback]:
        """Generate feedback on contradiction detection"""
        try:
            if hasattr(state, 'contradictions') and state.contradictions:
                contradiction_types = {}
                for contradiction in state.contradictions:
                    if hasattr(contradiction, 'contradiction_type'):
                        ctype = str(contradiction.contradiction_type)
                        contradiction_types[ctype] = contradiction_types.get(ctype, 0) + 1
                
                # Check for patterns in contradiction types
                if contradiction_types:
                    most_common = max(contradiction_types.items(), key=lambda x: x[1])
                    
                    return SystemFeedback(
                        feedback_type=FeedbackType.CONTRADICTION_DETECTION,
                        category="contradiction_detection",
                        description=f"Frequent contradiction type detected: {most_common[0]}",
                        details=f"Found {most_common[1]} instances of {most_common[0]}",
                        severity="low",
                        suggested_action=f"Review {most_common[0]} detection logic",
                        confidence=0.7
                    )
        except Exception as e:
            logger.warning(f"Contradiction feedback generation failed: {e}")
        
        return None
    
    def _generate_classification_feedback(self, state: ProcessingState) -> Optional[SystemFeedback]:
        """Generate feedback on document classification"""
        try:
            if hasattr(state, 'semantic_analysis') and state.semantic_analysis:
                doc_type = state.semantic_analysis.get('document_type', 'unknown')
                confidence = state.semantic_analysis.get('confidence', 0.5)
                
                # Check if classification seems uncertain
                if confidence < 0.4 and doc_type != 'unknown':
                    return SystemFeedback(
                        feedback_type=FeedbackType.DOCUMENT_CLASSIFICATION,
                        category="document_classification",
                        description="Low confidence in document classification",
                        details=f"Classified as '{doc_type}' with only {confidence:.0%} confidence",
                        severity="low",
                        suggested_action="Review document type classification criteria",
                        confidence=1 - confidence  # Lower classification confidence = higher feedback confidence
                    )
        except Exception as e:
            logger.warning(f"Classification feedback generation failed: {e}")
        
        return None
    
    def _generate_quality_feedback(self, state: ProcessingState) -> Optional[SystemFeedback]:
        """Generate feedback on quality assessment"""
        try:
            if hasattr(state, 'quality_scores') and state.quality_scores:
                quality_values = [score.overall for score in state.quality_scores.values()]
                avg_quality = sum(quality_values) / len(quality_values)
                
                # Check if quality assessment seems inconsistent with other metrics
                if avg_quality < 0.4:
                    # Document is low quality but may have good extraction
                    if hasattr(state, 'field_confidences') and state.field_confidences:
                        overall_confidence = state.field_confidences.get('overall', 0.5)
                        if overall_confidence > 0.7:
                            return SystemFeedback(
                                feedback_type=FeedbackType.QUALITY_ASSESSMENT,
                                category="quality_assessment",
                                description="Quality assessment may be too conservative",
                                details=f"Low quality score ({avg_quality:.0%}) but high confidence ({overall_confidence:.0%}) in results",
                                severity="low",
                                suggested_action="Review quality scoring impact on confidence",
                                confidence=0.6
                            )
        except Exception as e:
            logger.warning(f"Quality feedback generation failed: {e}")
        
        return None
    
    def _generate_risk_feedback(self, state: ProcessingState) -> Optional[SystemFeedback]:
        """Generate feedback on risk assessment"""
        try:
            if hasattr(state, 'risk_score'):
                risk_score = state.risk_score
                
                # Check if risk assessment aligns with other factors
                if risk_score > 0.7:
                    # High risk but may have mitigating factors
                    if hasattr(state, 'field_confidences') and state.field_confidences:
                        overall_confidence = state.field_confidences.get('overall', 0.5)
                        if overall_confidence > 0.8:
                            return SystemFeedback(
                                feedback_type=FeedbackType.RISK_ASSESSMENT,
                                category="risk_assessment",
                                description="Risk assessment may need calibration",
                                details=f"High risk score ({risk_score:.0%}) but high confidence ({overall_confidence:.0%}) in results",
                                severity="medium",
                                suggested_action="Review risk scoring algorithms",
                                confidence=0.5
                            )
        except Exception as e:
            logger.warning(f"Risk feedback generation failed: {e}")
        
        return None
    
    def _analyze_feedback_for_improvements(self, feedback_items: List[SystemFeedback]) -> List[Dict[str, Any]]:
        """Analyze feedback to suggest system improvements"""
        improvements = []
        
        # Group feedback by category
        feedback_by_category = {}
        for feedback in feedback_items:
            category = feedback.category
            if category not in feedback_by_category:
                feedback_by_category[category] = []
            feedback_by_category[category].append(feedback)
        
        # Generate improvement suggestions
        for category, items in feedback_by_category.items():
            if len(items) >= 2:  # Multiple feedback items in same category
                improvement = {
                    "category": category,
                    "description": f"Multiple feedback items in {self.feedback_categories.get(category, category)}",
                    "priority": "medium",
                    "affected_components": self._get_affected_components(category),
                    "suggested_changes": self._get_suggested_changes(category, items)
                }
                improvements.append(improvement)
        
        return improvements
    
    def _get_affected_components(self, category: str) -> List[str]:
        """Get components affected by feedback category"""
        component_map = {
            "confidence_calibration": ["ConfidenceArbitrationAgent", "OCRReliabilityAgent"],
            "entity_extraction": ["EntityIntelligenceAgent"],
            "contradiction_detection": ["ContradictionDetectionAgent"],
            "document_classification": ["SemanticReasoningAgent"],
            "quality_assessment": ["LayoutStrategyAgent", "VisualElementDetector"],
            "risk_assessment": ["RiskComplianceAgent"]
        }
        return component_map.get(category, ["Unknown"])
    
    def _get_suggested_changes(self, category: str, feedback_items: List[SystemFeedback]) -> List[str]:
        """Get suggested changes based on feedback"""
        change_map = {
            "confidence_calibration": [
                "Review confidence weighting parameters",
                "Add more confidence calibration factors",
                "Implement dynamic confidence adjustment"
            ],
            "entity_extraction": [
                "Improve entity context detection",
                "Add more entity normalization rules",
                "Implement entity validation checks"
            ],
            "contradiction_detection": [
                "Add more contradiction detection rules",
                "Improve contradiction severity scoring",
                "Implement contradiction resolution suggestions"
            ],
            "document_classification": [
                "Add more document type indicators",
                "Improve classification confidence calculation",
                "Implement hierarchical classification"
            ],
            "quality_assessment": [
                "Refine quality scoring metrics",
                "Add more quality assessment factors",
                "Implement quality-based processing adjustment"
            ],
            "risk_assessment": [
                "Review risk factor weights",
                "Add more risk assessment criteria",
                "Implement risk mitigation suggestions"
            ]
        }
        return change_map.get(category, ["Review and optimize component"])
    
    def _store_feedback(self, document_id: str, feedback_items: List[SystemFeedback]):
        """Store feedback for system learning"""
        try:
            feedback_record = {
                "document_id": document_id,
                "timestamp": datetime.now().isoformat(),
                "feedback_count": len(feedback_items),
                "feedback_items": [
                    {
                        "type": str(item.feedback_type),
                        "category": item.category,
                        "description": item.description,
                        "severity": item.severity,
                        "confidence": item.confidence
                    }
                    for item in feedback_items
                ]
            }
            
            self.feedback_history.append(feedback_record)
            
            # Keep only recent history (in production, this would be in database)
            if len(self.feedback_history) > 100:
                self.feedback_history = self.feedback_history[-50:]
                
            logger.debug(f"Stored feedback for {document_id}")
            
        except Exception as e:
            logger.warning(f"Feedback storage failed: {e}")