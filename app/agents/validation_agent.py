from typing import Dict, Any, List
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ValidationAgent:
    """Simple Validation Agent for quality checks"""
    
    async def validate_document(self, document_id: str, 
                              fused_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate document results
        
        Args:
            document_id: Document ID
            fused_results: Results from fusion agent
            
        Returns:
            Validation results
        """
        try:
            logger.info(f"Validating document {document_id}")
            
            # Get structured output
            structured_output = fused_results.get("structured_output", {})
            fields = structured_output.get("fields", {})
            
            # Simple validation checks
            flags = []
            recommendations = []
            
            # Check for empty fields
            empty_fields = []
            for field_name, field_data in fields.items():
                value = field_data.get("value", "")
                if not value or str(value).strip() == "":
                    empty_fields.append(field_name)
            
            if empty_fields:
                flags.append({
                    "type": "EMPTY_FIELD",
                    "fields": empty_fields,
                    "severity": "medium"
                })
                recommendations.append(f"Check empty fields: {', '.join(empty_fields)}")
            
            # Check confidence scores
            low_confidence_fields = []
            for field_name, field_data in fields.items():
                confidence = field_data.get("confidence", 0.0)
                if confidence < 0.5:
                    low_confidence_fields.append(field_name)
            
            if low_confidence_fields:
                flags.append({
                    "type": "LOW_CONFIDENCE",
                    "fields": low_confidence_fields,
                    "severity": "low"
                })
                recommendations.append(f"Verify low confidence fields: {', '.join(low_confidence_fields)}")
            
            # Count total fields
            total_fields = len(fields)
            filled_fields = sum(1 for field_data in fields.values() 
                              if field_data.get("value", ""))
            
            completeness_score = filled_fields / total_fields if total_fields > 0 else 0
            
            validation_summary = {
                "total_fields": total_fields,
                "filled_fields": filled_fields,
                "completeness_score": completeness_score,
                "status": "PASS" if completeness_score > 0.7 else "REVIEW_NEEDED"
            }
            
            return {
                "success": True,
                "document_id": document_id,
                "validation_summary": validation_summary,
                "flags": flags,
                "recommendations": recommendations,
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "success": False,
                "document_id": document_id,
                "validation_summary": {"status": "FAILED"},
                "flags": [],
                "recommendations": ["Validation error occurred"],
                "errors": [str(e)]
            }