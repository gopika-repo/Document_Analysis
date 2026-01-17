from typing import Dict, List, Any
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ExplanationAgent:
    """Agent for generating human-readable explanations of processing results"""
    
    def __init__(self):
        self.explanation_templates = {
            "high_confidence": "The system has high confidence in these results because multiple sources agree.",
            "medium_confidence": "Results are moderately confident. Some cross-verification is available.",
            "low_confidence": "Results have low confidence. Manual verification is recommended.",
            "contradiction_found": "A contradiction was found: {details}",
            "risk_identified": "Risk identified: {risk_factor}. Recommendation: {recommendation}",
            "compliance_issue": "Compliance issue: {issue}. Required: {requirement}",
            "quality_issue": "Document quality affects confidence: {issue}",
            "success_summary": "Document processing completed successfully with {confidence}% confidence."
        }
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Generate explanations for processing results"""
        try:
            logger.info(f"Generating explanations for {state.document_id}")
            
            explanations = {}
            
            # Generate confidence explanations
            explanations.update(self._explain_confidence(state))
            
            # Generate contradiction explanations
            explanations.update(self._explain_contradictions(state))
            
            # Generate risk explanations
            explanations.update(self._explain_risks(state))
            
            # Generate compliance explanations
            explanations.update(self._explain_compliance(state))
            
            # Generate quality explanations
            explanations.update(self._explain_quality(state))
            
            # Generate summary explanation
            explanations["summary"] = self._generate_summary_explanation(state)
            
            state.explanations = explanations
            logger.info(f"Generated {len(explanations)} explanations")
            
            return state
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            state.explanations = {"error": f"Explanation generation failed: {str(e)}"}
            return state
    
    def _explain_confidence(self, state: ProcessingState) -> Dict[str, str]:
        """Generate explanations for confidence scores"""
        explanations = {}
        
        try:
            if hasattr(state, 'field_confidences') and state.field_confidences:
                overall_confidence = state.field_confidences.get('overall', 0.0)
                
                if overall_confidence > 0.7:
                    explanations["overall_confidence"] = self.explanation_templates["high_confidence"]
                elif overall_confidence > 0.4:
                    explanations["overall_confidence"] = self.explanation_templates["medium_confidence"]
                else:
                    explanations["overall_confidence"] = self.explanation_templates["low_confidence"]
                
                # Explain individual confidence factors
                confidence_factors = []
                for field, score in state.field_confidences.items():
                    if field != "overall" and score > 0:
                        if score > 0.7:
                            level = "high"
                        elif score > 0.4:
                            level = "medium"
                        else:
                            level = "low"
                        
                        confidence_factors.append(f"{field.replace('_', ' ')}: {level} confidence ({score:.0%})")
                
                if confidence_factors:
                    explanations["confidence_breakdown"] = "Confidence breakdown: " + "; ".join(confidence_factors)
        
        except Exception as e:
            logger.warning(f"Confidence explanation failed: {e}")
        
        return explanations
    
    def _explain_contradictions(self, state: ProcessingState) -> Dict[str, str]:
        """Generate explanations for contradictions"""
        explanations = {}
        
        try:
            if hasattr(state, 'contradictions') and state.contradictions:
                contradiction_count = len(state.contradictions)
                explanations["contradiction_count"] = f"Found {contradiction_count} contradiction(s) in the document."
                
                # Explain each high/medium severity contradiction
                important_contradictions = []
                for i, contradiction in enumerate(state.contradictions[:3]):  # Limit to first 3
                    if hasattr(contradiction, 'severity'):
                        severity = str(contradiction.severity).lower()
                        if severity in ['high', 'medium', 'critical']:
                            exp = self.explanation_templates["contradiction_found"].format(
                                details=f"{contradiction.explanation} (Confidence: {contradiction.confidence:.0%})"
                            )
                            important_contradictions.append(exp)
                
                if important_contradictions:
                    explanations["contradiction_details"] = "Key contradictions: " + " | ".join(important_contradictions)
            else:
                explanations["contradiction_status"] = "No significant contradictions found."
        
        except Exception as e:
            logger.warning(f"Contradiction explanation failed: {e}")
        
        return explanations
    
    def _explain_risks(self, state: ProcessingState) -> Dict[str, str]:
        """Generate explanations for risks"""
        explanations = {}
        
        try:
            if hasattr(state, 'risk_score'):
                risk_score = state.risk_score
                
                if risk_score > 0.7:
                    risk_level = "HIGH"
                elif risk_score > 0.4:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
                
                explanations["risk_level"] = f"Document risk level: {risk_level} ({risk_score:.0%})"
                
                # Explain risk factors
                if hasattr(state, 'compliance_issues') and state.compliance_issues:
                    high_risk_issues = [issue for issue in state.compliance_issues 
                                      if issue.get('severity') == 'high']
                    if high_risk_issues:
                        explanations["risk_factors"] = f"High-risk issues: {len(high_risk_issues)} critical compliance gaps."
        
        except Exception as e:
            logger.warning(f"Risk explanation failed: {e}")
        
        return explanations
    
    def _explain_compliance(self, state: ProcessingState) -> Dict[str, str]:
        """Generate explanations for compliance issues"""
        explanations = {}
        
        try:
            if hasattr(state, 'compliance_issues') and state.compliance_issues:
                issue_count = len(state.compliance_issues)
                explanations["compliance_status"] = f"Found {issue_count} compliance issue(s)."
                
                # List critical issues
                critical_issues = []
                for issue in state.compliance_issues:
                    if issue.get('severity') == 'high':
                        exp = self.explanation_templates["compliance_issue"].format(
                            issue=issue.get('issue', 'unknown'),
                            requirement=issue.get('requirement', 'N/A')
                        )
                        critical_issues.append(exp)
                
                if critical_issues:
                    explanations["critical_compliance"] = "Critical compliance gaps: " + " | ".join(critical_issues[:2])
            else:
                explanations["compliance_status"] = "No significant compliance issues detected."
        
        except Exception as e:
            logger.warning(f"Compliance explanation failed: {e}")
        
        return explanations
    
    def _explain_quality(self, state: ProcessingState) -> Dict[str, str]:
        """Generate explanations for quality issues"""
        explanations = {}
        
        try:
            if hasattr(state, 'quality_scores') and state.quality_scores:
                # Calculate average quality
                quality_scores = [score.overall for score in state.quality_scores.values()]
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
                
                if avg_quality < 0.6:
                    explanations["document_quality"] = self.explanation_templates["quality_issue"].format(
                        issue=f"Average quality score: {avg_quality:.0%}. This may affect extraction accuracy."
                    )
                else:
                    explanations["document_quality"] = f"Document quality is good ({avg_quality:.0%})."
        
        except Exception as e:
            logger.warning(f"Quality explanation failed: {e}")
        
        return explanations
    
    def _generate_summary_explanation(self, state: ProcessingState) -> str:
        """Generate overall summary explanation"""
        try:
            # Gather key metrics
            confidence = state.field_confidences.get('overall', 0.5) if hasattr(state, 'field_confidences') else 0.5
            contradiction_count = len(state.contradictions) if hasattr(state, 'contradictions') else 0
            risk_score = state.risk_score if hasattr(state, 'risk_score') else 0.3
            entity_count = sum(len(entities) for entities in state.extracted_entities.values()) if hasattr(state, 'extracted_entities') else 0
            
            # Build summary
            summary_parts = []
            
            # Confidence part
            if confidence > 0.7:
                conf_text = "high confidence"
            elif confidence > 0.4:
                conf_text = "moderate confidence"
            else:
                conf_text = "low confidence"
            
            summary_parts.append(f"Processing completed with {conf_text} ({confidence:.0%})")
            
            # Entity extraction part
            if entity_count > 0:
                summary_parts.append(f"Extracted {entity_count} entities")
            
            # Contradiction part
            if contradiction_count > 0:
                summary_parts.append(f"Found {contradiction_count} contradiction(s)")
            else:
                summary_parts.append("No contradictions detected")
            
            # Risk part
            if risk_score > 0.7:
                risk_text = "high risk"
            elif risk_score > 0.4:
                risk_text = "medium risk"
            else:
                risk_text = "low risk"
            
            summary_parts.append(f"Document has {risk_text} level")
            
            # Final summary
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.warning(f"Summary explanation failed: {e}")
            return "Document processing completed. Review detailed results for specific insights."