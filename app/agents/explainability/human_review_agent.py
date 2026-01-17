from typing import Dict, List, Any, Optional
from app.core.models import ProcessingState, ReviewPriority, ReviewItem
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class HumanReviewAgent:
    """Agent for identifying elements that need human review"""
    
    def __init__(self):
        self.priority_thresholds = {
            ReviewPriority.CRITICAL: 0.9,
            ReviewPriority.HIGH: 0.7,
            ReviewPriority.MEDIUM: 0.5,
            ReviewPriority.LOW: 0.3
        }
        
        self.review_categories = {
            "confidence": "Low confidence areas",
            "contradiction": "Contradictions found",
            "compliance": "Compliance issues",
            "risk": "High risk elements",
            "quality": "Document quality issues",
            "ambiguity": "Ambiguous elements"
        }
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Identify elements needing human review"""
        try:
            logger.info(f"Identifying review items for {state.document_id}")
            
            review_items = []
            
            # Find items based on different criteria
            review_items.extend(self._find_low_confidence_items(state))
            review_items.extend(self._find_contradiction_items(state))
            review_items.extend(self._find_compliance_items(state))
            review_items.extend(self._find_risk_items(state))
            review_items.extend(self._find_quality_items(state))
            review_items.extend(self._find_ambiguous_items(state))
            
            # Sort by priority
            review_items.sort(key=lambda x: self._priority_to_score(x.priority), reverse=True)
            
            # Limit to top items
            review_items = review_items[:10]
            
            state.review_items = review_items
            logger.info(f"Identified {len(review_items)} items for human review")
            
            return state
            
        except Exception as e:
            logger.error(f"Human review identification failed: {e}")
            state.review_items = []
            return state
    
    def _find_low_confidence_items(self, state: ProcessingState) -> List[ReviewItem]:
        """Find items with low confidence scores"""
        items = []
        
        try:
            if hasattr(state, 'field_confidences') and state.field_confidences:
                for field, confidence in state.field_confidences.items():
                    if field != "overall" and confidence < 0.5:
                        priority = self._determine_priority(1 - confidence)
                        items.append(ReviewItem(
                            category="confidence",
                            priority=priority,
                            description=f"Low confidence in {field.replace('_', ' ')}",
                            details=f"Confidence score: {confidence:.0%}",
                            suggested_action="Verify extracted information",
                            confidence=1 - confidence  # Lower confidence = higher need for review
                        ))
            
            # Check OCR confidence
            if hasattr(state, 'ocr_confidence') and state.ocr_confidence:
                for page_num, confidence in state.ocr_confidence.items():
                    if confidence < 0.6:
                        priority = self._determine_priority(1 - confidence)
                        items.append(ReviewItem(
                            category="confidence",
                            priority=priority,
                            description=f"Low OCR confidence on page {page_num + 1}",
                            details=f"OCR confidence: {confidence:.0%}",
                            suggested_action="Review page text for accuracy",
                            confidence=1 - confidence
                        ))
        
        except Exception as e:
            logger.warning(f"Low confidence item detection failed: {e}")
        
        return items
    
    def _find_contradiction_items(self, state: ProcessingState) -> List[ReviewItem]:
        """Find items with contradictions"""
        items = []
        
        try:
            if hasattr(state, 'contradictions') and state.contradictions:
                for i, contradiction in enumerate(state.contradictions):
                    if hasattr(contradiction, 'severity'):
                        severity_str = str(contradiction.severity)
                        if severity_str in ["HIGH", "CRITICAL", "MEDIUM"]:
                            priority = self._priority_from_severity(severity_str)
                            items.append(ReviewItem(
                                category="contradiction",
                                priority=priority,
                                description=f"Contradiction: {contradiction.field_a} vs {contradiction.field_b}",
                                details=contradiction.explanation,
                                suggested_action="Resolve contradiction",
                                confidence=contradiction.confidence if hasattr(contradiction, 'confidence') else 0.8
                            ))
        
        except Exception as e:
            logger.warning(f"Contradiction item detection failed: {e}")
        
        return items
    
    def _find_compliance_items(self, state: ProcessingState) -> List[ReviewItem]:
        """Find compliance-related items needing review"""
        items = []
        
        try:
            if hasattr(state, 'compliance_issues') and state.compliance_issues:
                for issue in state.compliance_issues:
                    severity = issue.get('severity', 'medium')
                    priority = self._priority_from_severity(severity.upper())
                    
                    items.append(ReviewItem(
                        category="compliance",
                        priority=priority,
                        description=f"Compliance issue: {issue.get('field', 'unknown')}",
                        details=f"{issue.get('issue', 'N/A')} - {issue.get('requirement', 'N/A')}",
                        suggested_action="Address compliance requirement",
                        confidence=0.9 if severity == 'high' else 0.7
                    ))
        
        except Exception as e:
            logger.warning(f"Compliance item detection failed: {e}")
        
        return items
    
    def _find_risk_items(self, state: ProcessingState) -> List[ReviewItem]:
        """Find high-risk items"""
        items = []
        
        try:
            if hasattr(state, 'risk_score'):
                risk_score = state.risk_score
                
                if risk_score > 0.7:
                    items.append(ReviewItem(
                        category="risk",
                        priority=ReviewPriority.CRITICAL,
                        description="High overall document risk",
                        details=f"Risk score: {risk_score:.0%}",
                        suggested_action="Comprehensive review required",
                        confidence=risk_score
                    ))
                
                # Check specific risk factors
                if hasattr(state, 'processing_metadata') and state.processing_metadata:
                    risk_assessment = state.processing_metadata.get('risk_assessment', {})
                    risk_factors = risk_assessment.get('risk_factors', [])
                    
                    for factor in risk_factors:
                        if factor.get('severity') == 'high':
                            items.append(ReviewItem(
                                category="risk",
                                priority=ReviewPriority.HIGH,
                                description=f"High risk: {factor.get('factor', 'unknown')}",
                                details=f"Severity: {factor.get('severity', 'N/A')}",
                                suggested_action="Address specific risk factor",
                                confidence=0.8
                            ))
        
        except Exception as e:
            logger.warning(f"Risk item detection failed: {e}")
        
        return items
    
    def _find_quality_items(self, state: ProcessingState) -> List[ReviewItem]:
        """Find items related to document quality"""
        items = []
        
        try:
            if hasattr(state, 'quality_scores') and state.quality_scores:
                for page_num, score in state.quality_scores.items():
                    if hasattr(score, 'overall') and score.overall < 0.6:
                        priority = ReviewPriority.MEDIUM if score.overall < 0.4 else ReviewPriority.LOW
                        items.append(ReviewItem(
                            category="quality",
                            priority=priority,
                            description=f"Poor document quality on page {page_num + 1}",
                            details=f"Quality score: {score.overall:.0%}",
                            suggested_action="Check scan/image quality",
                            confidence=1 - score.overall
                        ))
        
        except Exception as e:
            logger.warning(f"Quality item detection failed: {e}")
        
        return items
    
    def _find_ambiguous_items(self, state: ProcessingState) -> List[ReviewItem]:
        """Find ambiguous elements"""
        items = []
        
        try:
            # Check for ambiguous entities
            if hasattr(state, 'extracted_entities') and state.extracted_entities:
                for entity_type, entities in state.extracted_entities.items():
                    # Look for potentially ambiguous entities
                    if entity_type == "names" and len(entities) > 3:
                        items.append(ReviewItem(
                            category="ambiguity",
                            priority=ReviewPriority.LOW,
                            description=f"Multiple names extracted ({len(entities)})",
                            details=f"Names: {', '.join(entities[:3])}...",
                            suggested_action="Verify correct name identification",
                            confidence=0.5
                        ))
            
            # Check for uncertainty in semantic analysis
            if hasattr(state, 'semantic_analysis') and state.semantic_analysis:
                uncertainty = state.semantic_analysis.get('uncertainty_level', 0.0)
                if uncertainty > 0.3:
                    items.append(ReviewItem(
                        category="ambiguity",
                        priority=ReviewPriority.LOW,
                        description="High uncertainty in document interpretation",
                        details=f"Uncertainty level: {uncertainty:.0%}",
                        suggested_action="Review document context",
                        confidence=uncertainty
                    ))
        
        except Exception as e:
            logger.warning(f"Ambiguity item detection failed: {e}")
        
        return items
    
    def _determine_priority(self, score: float) -> ReviewPriority:
        """Determine review priority based on score"""
        if score >= self.priority_thresholds[ReviewPriority.CRITICAL]:
            return ReviewPriority.CRITICAL
        elif score >= self.priority_thresholds[ReviewPriority.HIGH]:
            return ReviewPriority.HIGH
        elif score >= self.priority_thresholds[ReviewPriority.MEDIUM]:
            return ReviewPriority.MEDIUM
        else:
            return ReviewPriority.LOW
    
    def _priority_from_severity(self, severity: str) -> ReviewPriority:
        """Convert severity string to priority"""
        severity_map = {
            "CRITICAL": ReviewPriority.CRITICAL,
            "HIGH": ReviewPriority.HIGH,
            "MEDIUM": ReviewPriority.MEDIUM,
            "LOW": ReviewPriority.LOW
        }
        return severity_map.get(severity.upper(), ReviewPriority.MEDIUM)
    
    def _priority_to_score(self, priority: ReviewPriority) -> int:
        """Convert priority to numeric score for sorting"""
        priority_scores = {
            ReviewPriority.CRITICAL: 4,
            ReviewPriority.HIGH: 3,
            ReviewPriority.MEDIUM: 2,
            ReviewPriority.LOW: 1
        }
        return priority_scores.get(priority, 0)