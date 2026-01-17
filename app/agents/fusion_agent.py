from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from app.utils.logger import setup_logger
import json

logger = setup_logger(__name__)

class ValidationState(BaseModel):
    """State for Validation Agent"""
    document_id: str
    fused_results: Dict[str, Any] = Field(default_factory=dict)
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    flags: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

class ValidationAgent:
    """Validation Agent for cross-checking and quality assurance"""
    
    def create_graph(self) -> StateGraph:
        """Create LangGraph for validation processing"""
        workflow = StateGraph(ValidationState)
        
        # Add nodes
        workflow.add_node("cross_check", self.cross_check)
        workflow.add_node("detect_inconsistencies", self.detect_inconsistencies)
        workflow.add_node("generate_flags", self.generate_flags)
        workflow.add_node("provide_recommendations", self.provide_recommendations)
        
        # Add edges
        workflow.add_edge("cross_check", "detect_inconsistencies")
        workflow.add_edge("detect_inconsistencies", "generate_flags")
        workflow.add_edge("generate_flags", "provide_recommendations")
        workflow.add_edge("provide_recommendations", END)
        
        # Set entry point
        workflow.set_entry_point("cross_check")
        
        return workflow
    
    def cross_check(self, state: ValidationState) -> ValidationState:
        """Cross-check extracted values"""
        try:
            logger.info(f"Cross-checking document {state.document_id}")
            
            validation_results = {
                "consistency_checks": [],
                "plausibility_checks": [],
                "completeness_checks": []
            }
            
            # Check field consistency
            fields = state.fused_results.get("structured_output", {}).get("fields", {})
            for field_name, field_data in fields.items():
                consistency = self._check_field_consistency(field_data)
                validation_results["consistency_checks"].append({
                    "field": field_name,
                    "status": consistency["status"],
                    "details": consistency["details"]
                })
            
            # Check plausibility
            plausibility = self._check_plausibility(fields)
            validation_results["plausibility_checks"] = plausibility
            
            # Check completeness
            completeness = self._check_completeness(fields)
            validation_results["completeness_checks"] = completeness
            
            state.validation_results = validation_results
            logger.info(f"Completed {len(validation_results['consistency_checks'])} checks")
            
        except Exception as e:
            error_msg = f"Cross-checking failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def _check_field_consistency(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency of a field"""
        confidence = field_data.get("confidence", 0.0)
        sources = field_data.get("source", "").split(" + ")
        
        if confidence < 0.5:
            return {
                "status": "LOW_CONFIDENCE",
                "details": f"Confidence score {confidence:.2f} is below threshold"
            }
        elif len(sources) == 1:
            return {
                "status": "SINGLE_SOURCE",
                "details": "Field extracted from single modality only"
            }
        else:
            return {
                "status": "CONSISTENT",
                "details": "Field validated across multiple modalities"
            }
    
    def _check_plausibility(self, fields: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check plausibility of field values"""
        plausibility_checks = []
        
        # Check for numeric fields with reasonable values
        for field_name, field_data in fields.items():
            value = field_data.get("value", "")
            
            # Check if value contains numbers
            if any(char.isdigit() for char in str(value)):
                # Extract numbers
                import re
                numbers = re.findall(r'\d+\.?\d*', str(value))
                
                if numbers:
                    for num in numbers:
                        num_float = float(num)
                        if "amount" in field_name.lower() or "total" in field_name.lower():
                            if num_float > 1000000:  # Unusually large amount
                                plausibility_checks.append({
                                    "field": field_name,
                                    "issue": "UNUSUALLY_LARGE_VALUE",
                                    "value": num_float,
                                    "threshold": 1000000
                                })
        
        return plausibility_checks
    
    def _check_completeness(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Check completeness of extraction"""
        required_fields = ["date", "amount", "signature", "table"]
        found_fields = []
        
        for field_name in fields.keys():
            for required in required_fields:
                if required in field_name.lower():
                    found_fields.append(required)
        
        missing_fields = [f for f in required_fields if f not in found_fields]
        
        return {
            "total_required": len(required_fields),
            "found": len(found_fields),
            "missing": missing_fields,
            "completeness_score": len(found_fields) / len(required_fields) if required_fields else 1.0
        }
    
    def detect_inconsistencies(self, state: ValidationState) -> ValidationState:
        """Detect inconsistencies in the data"""
        try:
            logger.info(f"Detecting inconsistencies for document {state.document_id}")
            
            inconsistencies = []
            fields = state.fused_results.get("structured_output", {}).get("fields", {})
            
            # Check for contradictory information
            for field_name, field_data in fields.items():
                if "chart" in field_name.lower():
                    # Check if chart data contradicts text
                    chart_inconsistency = self._check_chart_text_consistency(
                        field_data, fields
                    )
                    if chart_inconsistency:
                        inconsistencies.append(chart_inconsistency)
            
            # Check temporal consistency
            temporal_issues = self._check_temporal_consistency(fields)
            inconsistencies.extend(temporal_issues)
            
            # Check numeric consistency
            numeric_issues = self._check_numeric_consistency(fields)
            inconsistencies.extend(numeric_issues)
            
            state.validation_results["inconsistencies"] = inconsistencies
            logger.info(f"Found {len(inconsistencies)} inconsistencies")
            
        except Exception as e:
            error_msg = f"Inconsistency detection failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def _check_chart_text_consistency(self, chart_field: Dict[str, Any], 
                                     all_fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if chart data contradicts text"""
        # Look for text fields about trends or comparisons
        text_fields = [
            (name, data) for name, data in all_fields.items()
            if "text" in data.get("source", "").lower() 
            and "trend" in name.lower() or "comparison" in name.lower()
        ]
        
        if text_fields:
            # This would involve comparing chart analysis with text descriptions
            # For now, return a placeholder check
            return {
                "type": "chart_text_contradiction",
                "chart_field": list(chart_field.keys())[0] if chart_field else "unknown",
                "description": "Potential contradiction between chart data and text description",
                "severity": "medium"
            }
        
        return None
    
    def _check_temporal_consistency(self, fields: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check temporal consistency"""
        temporal_issues = []
        dates = []
        
        # Extract dates from fields
        for field_name, field_data in fields.items():
            if "date" in field_name.lower():
                dates.append({
                    "field": field_name,
                    "value": field_data.get("value", "")
                })
        
        # Check if dates are in chronological order
        if len(dates) > 1:
            # This would parse dates and check order
            temporal_issues.append({
                "type": "multiple_dates_found",
                "dates": dates,
                "check_needed": "Chronological order verification"
            })
        
        return temporal_issues
    
    def _check_numeric_consistency(self, fields: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check numeric consistency"""
        numeric_issues = []
        amounts = []
        
        # Extract amounts from fields
        for field_name, field_data in fields.items():
            if "amount" in field_name.lower() or "total" in field_name.lower():
                value = field_data.get("value", "")
                # Extract numbers
                import re
                numbers = re.findall(r'\d+\.?\d*', str(value))
                if numbers:
                    amounts.append({
                        "field": field_name,
                        "value": float(numbers[0])
                    })
        
        # Check if amounts are consistent
        if len(amounts) > 1:
            # Check for significant differences
            values = [a["value"] for a in amounts]
            avg_value = sum(values) / len(values)
            
            for amount in amounts:
                deviation = abs(amount["value"] - avg_value) / avg_value if avg_value != 0 else 0
                if deviation > 0.5:  # More than 50% deviation
                    numeric_issues.append({
                        "type": "amount_inconsistency",
                        "field": amount["field"],
                        "value": amount["value"],
                        "average": avg_value,
                        "deviation": f"{deviation:.1%}"
                    })
        
        return numeric_issues
    
    def generate_flags(self, state: ValidationState) -> ValidationState:
        """Generate validation flags"""
        try:
            logger.info(f"Generating flags for document {state.document_id}")
            
            flags = []
            
            # Add flags from consistency checks
            for check in state.validation_results.get("consistency_checks", []):
                if check["status"] != "CONSISTENT":
                    flags.append({
                        "type": "CONSISTENCY_FLAG",
                        "field": check.get("field", "unknown"),
                        "status": check["status"],
                        "reason": check["details"],
                        "priority": "HIGH" if check["status"] == "LOW_CONFIDENCE" else "MEDIUM"
                    })
            
            # Add flags from plausibility checks
            for check in state.validation_results.get("plausibility_checks", []):
                flags.append({
                    "type": "PLAUSIBILITY_FLAG",
                    "field": check.get("field", "unknown"),
                    "status": "IMPLAUSIBLE_VALUE",
                    "reason": f"{check['issue']}: {check['value']} exceeds threshold {check['threshold']}",
                    "priority": "MEDIUM"
                })
            
            # Add flags from inconsistencies
            for inconsistency in state.validation_results.get("inconsistencies", []):
                flags.append({
                    "type": "INCONSISTENCY_FLAG",
                    "field": inconsistency.get("chart_field", "multiple"),
                    "status": "CONTRADICTION_DETECTED",
                    "reason": inconsistency["description"],
                    "priority": inconsistency.get("severity", "MEDIUM").upper()
                })
            
            # Add completeness flag if needed
            completeness = state.validation_results.get("completeness_checks", {})
            if completeness.get("completeness_score", 1.0) < 0.7:
                flags.append({
                    "type": "COMPLETENESS_FLAG",
                    "field": "document",
                    "status": "INCOMPLETE_EXTRACTION",
                    "reason": f"Only {completeness.get('found', 0)} of {completeness.get('total_required', 0)} required fields found",
                    "priority": "LOW"
                })
            
            state.flags = flags
            logger.info(f"Generated {len(flags)} validation flags")
            
        except Exception as e:
            error_msg = f"Flag generation failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def provide_recommendations(self, state: ValidationState) -> ValidationState:
        """Provide recommendations based on validation results"""
        try:
            logger.info(f"Providing recommendations for document {state.document_id}")
            
            recommendations = []
            
            # Recommendations based on flags
            for flag in state.flags:
                if flag["priority"] == "HIGH":
                    recommendations.append(
                        f"Manual review required for {flag['field']}: {flag['reason']}"
                    )
                elif flag["priority"] == "MEDIUM":
                    recommendations.append(
                        f"Consider verifying {flag['field']}: {flag['reason']}"
                    )
            
            # General recommendations
            if not state.flags:
                recommendations.append(
                    "Document validation passed all checks. No manual review needed."
                )
            elif len([f for f in state.flags if f["priority"] == "HIGH"]) == 0:
                recommendations.append(
                    "Document has minor issues. Consider batch review if multiple similar documents."
                )
            else:
                recommendations.append(
                    "Document has critical issues requiring immediate manual review."
                )
            
            # Add processing recommendations
            completeness = state.validation_results.get("completeness_checks", {})
            if completeness.get("missing"):
                recommendations.append(
                    f"Consider reprocessing to extract missing fields: {', '.join(completeness['missing'])}"
                )
            
            state.recommendations = recommendations
            logger.info(f"Generated {len(recommendations)} recommendations")
            
        except Exception as e:
            error_msg = f"Recommendation generation failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def validate_document(self, document_id: str,
                               fused_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document through validation pipeline"""
        try:
            # Initialize state
            state = ValidationState(
                document_id=document_id,
                fused_results=fused_results
            )
            
            # Create and run graph
            graph = self.create_graph()
            compiled_graph = graph.compile()
            
            # Execute graph
            result_state = compiled_graph.invoke(state)
            
            # Prepare response
            response = {
                "success": len(result_state.errors) == 0,
                "document_id": result_state.document_id,
                "validation_summary": {
                    "total_checks": len(result_state.validation_results.get("consistency_checks", [])),
                    "flags_generated": len(result_state.flags),
                    "overall_status": "PASS" if not result_state.flags else 
                                     "REVIEW_NEEDED" if any(f["priority"] == "HIGH" for f in result_state.flags) 
                                     else "WARNING"
                },
                "flags": result_state.flags,
                "recommendations": result_state.recommendations,
                "errors": result_state.errors
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "success": False,
                "document_id": document_id,
                "error": str(e)
            }