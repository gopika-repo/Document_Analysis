from typing import Dict, List, Any
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class RiskComplianceAgent:
    """Agent for assessing business risk and compliance issues"""
    
    def __init__(self):
        self.risk_factors = {
            "missing_signature": 0.8,
            "missing_date": 0.6,
            "missing_amount": 0.7,
            "contradictions_present": 0.9,
            "low_confidence": 0.5,
            "incomplete_information": 0.4
        }
        
        self.compliance_checks = {
            "financial": ["amount", "date", "signature", "company_name"],
            "legal": ["signature", "date", "parties", "terms"],
            "contract": ["effective_date", "termination_date", "signatures", "parties"]
        }
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Assess business risk and compliance"""
        try:
            logger.info(f"Assessing risk and compliance for {state.document_id}")
            
            risk_assessment = {
                "risk_score": 0.0,
                "risk_factors": [],
                "compliance_issues": [],
                "recommendations": [],
                "audit_readiness": 0.0
            }
            
            # Calculate risk score
            risk_score, risk_factors = self._calculate_risk_score(state)
            risk_assessment["risk_score"] = risk_score
            risk_assessment["risk_factors"] = risk_factors
            
            # Check compliance based on document type
            compliance_issues = self._check_compliance(state)
            risk_assessment["compliance_issues"] = compliance_issues
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_score, risk_factors, compliance_issues)
            risk_assessment["recommendations"] = recommendations
            
            # Calculate audit readiness
            audit_readiness = self._calculate_audit_readiness(risk_score, compliance_issues)
            risk_assessment["audit_readiness"] = audit_readiness
            
            state.risk_score = risk_score
            state.compliance_issues = compliance_issues
            state.processing_metadata = state.processing_metadata or {}
            state.processing_metadata["risk_assessment"] = risk_assessment
            
            logger.info(f"Risk assessment completed: score {risk_score:.2f}, audit readiness {audit_readiness:.2f}")
            
            return state
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            state.risk_score = 0.5  # Medium risk by default
            return state
    
    def _calculate_risk_score(self, state: ProcessingState) -> tuple:
        """Calculate overall risk score"""
        risk_factors = []
        total_weight = 0
        
        # Check for missing critical elements
        if self._has_missing_signature(state):
            risk_factors.append({
                "factor": "missing_signature",
                "severity": "high",
                "weight": self.risk_factors["missing_signature"]
            })
            total_weight += self.risk_factors["missing_signature"]
        
        if self._has_missing_date(state):
            risk_factors.append({
                "factor": "missing_date",
                "severity": "medium",
                "weight": self.risk_factors["missing_date"]
            })
            total_weight += self.risk_factors["missing_date"]
        
        if self._has_missing_amount(state):
            risk_factors.append({
                "factor": "missing_amount",
                "severity": "high",
                "weight": self.risk_factors["missing_amount"]
            })
            total_weight += self.risk_factors["missing_amount"]
        
        # Check for contradictions
        if hasattr(state, 'contradictions') and state.contradictions:
            high_severity_contradictions = [
                c for c in state.contradictions 
                if hasattr(c, 'severity') and str(c.severity).lower() in ['high', 'critical']
            ]
            
            if high_severity_contradictions:
                risk_factors.append({
                    "factor": "contradictions_present",
                    "severity": "high",
                    "weight": self.risk_factors["contradictions_present"],
                    "count": len(high_severity_contradictions)
                })
                total_weight += self.risk_factors["contradictions_present"]
        
        # Check for low confidence in processing
        if hasattr(state, 'field_confidences') and state.field_confidences:
            overall_confidence = state.field_confidences.get('overall', 0.5)
            if overall_confidence < 0.6:
                risk_factors.append({
                    "factor": "low_confidence",
                    "severity": "medium",
                    "weight": self.risk_factors["low_confidence"] * (1 - overall_confidence)
                })
                total_weight += self.risk_factors["low_confidence"] * (1 - overall_confidence)
        
        # Check for incomplete information
        if self._has_incomplete_information(state):
            risk_factors.append({
                "factor": "incomplete_information",
                "severity": "medium",
                "weight": self.risk_factors["incomplete_information"]
            })
            total_weight += self.risk_factors["incomplete_information"]
        
        # Calculate risk score (0-1, higher = more risk)
        if risk_factors:
            weighted_sum = sum(factor["weight"] for factor in risk_factors)
            risk_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            risk_score = 0.1  # Low risk if no factors found
        
        return min(risk_score, 1.0), risk_factors
    
    def _has_missing_signature(self, state: ProcessingState) -> bool:
        """Check if document is missing signature"""
        try:
            # Check visual elements for signatures
            if hasattr(state, 'visual_elements') and state.visual_elements:
                for elements in state.visual_elements.values():
                    for element in elements:
                        if hasattr(element, 'element_type') and element.element_type == 'signature':
                            return False
            
            # Check signature verification results
            if hasattr(state, 'signature_verification') and state.signature_verification:
                if isinstance(state.signature_verification, dict):
                    signatures_found = state.signature_verification.get('signatures_found', 0)
                    if signatures_found > 0:
                        return False
            
            return True  # No signature found
        
        except:
            return True  # Assume missing if check fails
    
    def _has_missing_date(self, state: ProcessingState) -> bool:
        """Check if document is missing date"""
        try:
            if hasattr(state, 'extracted_entities') and state.extracted_entities:
                dates = state.extracted_entities.get('dates', [])
                if dates:
                    return False
            
            return True
        
        except:
            return True
    
    def _has_missing_amount(self, state: ProcessingState) -> bool:
        """Check if document is missing monetary amount"""
        try:
            if hasattr(state, 'extracted_entities') and state.extracted_entities:
                amounts = state.extracted_entities.get('amounts', [])
                if amounts:
                    return False
            
            return True
        
        except:
            return True
    
    def _has_incomplete_information(self, state: ProcessingState) -> bool:
        """Check if document has incomplete information"""
        try:
            # Check if we have very little extracted data
            data_points = 0
            
            if hasattr(state, 'extracted_entities'):
                for entities in state.extracted_entities.values():
                    data_points += len(entities)
            
            if hasattr(state, 'visual_elements'):
                for elements in state.visual_elements.values():
                    data_points += len(elements)
            
            return data_points < 3  # Very few data points
        
        except:
            return True
    
    def _check_compliance(self, state: ProcessingState) -> List[Dict[str, Any]]:
        """Check compliance based on document type"""
        compliance_issues = []
        
        try:
            # Determine document type
            doc_type = "unknown"
            if hasattr(state, 'semantic_analysis') and state.semantic_analysis:
                doc_type = state.semantic_analysis.get('document_type', 'unknown')
            
            # Get required fields for this document type
            required_fields = self.compliance_checks.get(doc_type, [])
            
            if not required_fields:
                # Generic compliance check
                required_fields = ["date", "signature", "amount"]
            
            # Check each required field
            for field in required_fields:
                if field == "date" and self._has_missing_date(state):
                    compliance_issues.append({
                        "field": "date",
                        "issue": "missing_date",
                        "severity": "medium",
                        "requirement": "Documents should include a date"
                    })
                
                elif field == "signature" and self._has_missing_signature(state):
                    compliance_issues.append({
                        "field": "signature",
                        "issue": "missing_signature",
                        "severity": "high",
                        "requirement": "Documents should be signed"
                    })
                
                elif field == "amount" and self._has_missing_amount(state):
                    compliance_issues.append({
                        "field": "amount",
                        "issue": "missing_amount",
                        "severity": "high",
                        "requirement": "Financial documents should include amounts"
                    })
                
                elif field == "company_name":
                    if hasattr(state, 'extracted_entities') and state.extracted_entities:
                        companies = state.extracted_entities.get('companies', [])
                        if not companies:
                            compliance_issues.append({
                                "field": "company_name",
                                "issue": "missing_company",
                                "severity": "medium",
                                "requirement": "Business documents should include company names"
                            })
        
        except Exception as e:
            logger.warning(f"Compliance check failed: {e}")
            compliance_issues.append({
                "field": "system",
                "issue": "check_failed",
                "severity": "low",
                "requirement": f"Compliance check error: {str(e)}"
            })
        
        return compliance_issues
    
    def _generate_recommendations(self, risk_score: float, 
                                 risk_factors: List[Dict], 
                                 compliance_issues: List[Dict]) -> List[str]:
        """Generate recommendations based on risk assessment"""
        recommendations = []
        
        # High risk recommendations
        if risk_score > 0.7:
            recommendations.append("HIGH RISK: Manual review required immediately")
            recommendations.append("Consider consulting with subject matter expert")
        
        # Medium risk recommendations
        elif risk_score > 0.4:
            recommendations.append("MEDIUM RISK: Review recommended within 48 hours")
        
        # Specific recommendations based on risk factors
        for factor in risk_factors:
            if factor["factor"] == "missing_signature":
                recommendations.append("Add signature verification or obtain signed copy")
            elif factor["factor"] == "missing_date":
                recommendations.append("Document should include a clear date")
            elif factor["factor"] == "missing_amount":
                recommendations.append("Verify that all monetary amounts are present and correct")
            elif factor["factor"] == "contradictions_present":
                recommendations.append("Resolve identified contradictions before proceeding")
        
        # Compliance recommendations
        for issue in compliance_issues:
            if issue["severity"] == "high":
                recommendations.append(f"CRITICAL: Missing {issue['field']} - {issue['requirement']}")
            elif issue["severity"] == "medium":
                recommendations.append(f"Important: Address missing {issue['field']}")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Document appears to be in good order")
            recommendations.append("Standard processing can proceed")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_audit_readiness(self, risk_score: float, 
                                  compliance_issues: List[Dict]) -> float:
        """Calculate audit readiness score (0-1, higher = more ready)"""
        # Start with base score
        readiness_score = 1.0 - risk_score  # Inverse of risk
        
        # Apply penalties for compliance issues
        penalty = 0.0
        for issue in compliance_issues:
            if issue["severity"] == "high":
                penalty += 0.3
            elif issue["severity"] == "medium":
                penalty += 0.15
            elif issue["severity"] == "low":
                penalty += 0.05
        
        readiness_score -= penalty
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, readiness_score))