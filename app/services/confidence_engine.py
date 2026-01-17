from typing import Dict, List, Any, Optional
import numpy as np
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ConfidenceEngine:
    """Engine for computing confidence scores"""
    
    def __init__(self):
        # Weight configurations
        self.weights = {
            "ocr_confidence": 0.25,
            "visual_confidence": 0.25,
            "alignment_confidence": 0.20,
            "consistency_score": 0.15,
            "plausibility_score": 0.10,
            "completeness_score": 0.05
        }
        
        # Thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.6
        self.low_confidence_threshold = 0.4
    
    def compute_confidence(self,
                          visual_confidence: float = 0.0,
                          text_confidence: float = 0.0,
                          alignment_confidence: float = 0.0,
                          consistency_score: float = 1.0,
                          plausibility_score: float = 1.0,
                          completeness_score: float = 1.0,
                          conflicts: Optional[List[Dict[str, Any]]] = None) -> float:
        """
        Compute overall confidence score
        
        Args:
            visual_confidence: Confidence from visual detection
            text_confidence: Confidence from OCR/text extraction
            alignment_confidence: Confidence in modality alignment
            consistency_score: Consistency score (0-1)
            plausibility_score: Plausibility score (0-1)
            completeness_score: Completeness score (0-1)
            conflicts: List of conflicts affecting confidence
            
        Returns:
            Overall confidence score (0-1)
        """
        try:
            # Adjust for conflicts
            conflict_penalty = self._calculate_conflict_penalty(conflicts)
            
            # Calculate weighted average
            weighted_sum = (
                visual_confidence * self.weights["visual_confidence"] +
                text_confidence * self.weights["ocr_confidence"] +
                alignment_confidence * self.weights["alignment_confidence"] +
                consistency_score * self.weights["consistency_score"] +
                plausibility_score * self.weights["plausibility_score"] +
                completeness_score * self.weights["completeness_score"]
            )
            
            # Apply conflict penalty
            final_confidence = weighted_sum * conflict_penalty
            
            # Ensure within bounds
            final_confidence = max(0.0, min(1.0, final_confidence))
            
            logger.debug(f"Computed confidence: {final_confidence:.3f}")
            return final_confidence
            
        except Exception as e:
            logger.error(f"Confidence computation failed: {e}")
            return 0.0
    
    def _calculate_conflict_penalty(self, 
                                   conflicts: Optional[List[Dict[str, Any]]]) -> float:
        """Calculate penalty based on conflicts"""
        if not conflicts:
            return 1.0
        
        # Severity weights
        severity_weights = {
            "HIGH": 0.6,
            "MEDIUM": 0.3,
            "LOW": 0.1
        }
        
        total_penalty = 0.0
        for conflict in conflicts:
            severity = conflict.get("severity", "MEDIUM").upper()
            penalty = severity_weights.get(severity, 0.3)
            total_penalty += penalty
        
        # Apply diminishing returns for multiple conflicts
        if total_penalty > 0:
            penalty_factor = 1.0 / (1.0 + total_penalty)
        else:
            penalty_factor = 1.0
        
        return penalty_factor
    
    def compute_field_confidence(self, 
                               field_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute confidence for a specific field
        
        Args:
            field_data: Field data with confidence indicators
            
        Returns:
            Confidence analysis
        """
        try:
            # Extract confidence indicators
            indicators = self._extract_confidence_indicators(field_data)
            
            # Calculate individual scores
            scores = {
                "source_agreement": self._calculate_source_agreement(
                    indicators.get("sources", [])
                ),
                "value_consistency": self._calculate_value_consistency(
                    indicators.get("values", [])
                ),
                "context_relevance": indicators.get("context_relevance", 0.5),
                "historical_accuracy": indicators.get("historical_accuracy", 0.5)
            }
            
            # Calculate overall field confidence
            field_weights = {
                "source_agreement": 0.4,
                "value_consistency": 0.3,
                "context_relevance": 0.2,
                "historical_accuracy": 0.1
            }
            
            overall_score = sum(
                scores[key] * field_weights[key] 
                for key in field_weights.keys()
            )
            
            # Determine confidence level
            confidence_level = self._get_confidence_level(overall_score)
            
            return {
                "overall_confidence": overall_score,
                "confidence_level": confidence_level,
                "component_scores": scores,
                "indicators": indicators
            }
            
        except Exception as e:
            logger.error(f"Field confidence computation failed: {e}")
            return {
                "overall_confidence": 0.0,
                "confidence_level": "LOW",
                "error": str(e)
            }
    
    def _extract_confidence_indicators(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract confidence indicators from field data"""
        indicators = {
            "sources": [],
            "values": [],
            "context_relevance": 0.5,
            "historical_accuracy": 0.5
        }
        
        # Extract sources
        if "source" in field_data:
            sources = field_data["source"].split(" + ")
            indicators["sources"] = sources
        
        # Extract values from different sources
        if "visual_evidence" in field_data:
            indicators["values"].append({
                "source": "visual",
                "value": field_data["visual_evidence"],
                "confidence": field_data.get("visual_confidence", 0.5)
            })
        
        if "text_evidence" in field_data:
            indicators["values"].append({
                "source": "text",
                "value": field_data["text_evidence"],
                "confidence": field_data.get("text_confidence", 0.5)
            })
        
        # Context relevance (simplified)
        if "context_matches" in field_data:
            indicators["context_relevance"] = field_data["context_matches"]
        
        return indicators
    
    def _calculate_source_agreement(self, sources: List[str]) -> float:
        """Calculate agreement between sources"""
        if not sources:
            return 0.3  # Low confidence for no sources
        
        if len(sources) == 1:
            return 0.6  # Medium confidence for single source
        
        # Multiple sources increase confidence
        base_score = 0.7
        
        # Bonus for specific source combinations
        if "OCR" in sources and "Visual Detection" in sources:
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def _calculate_value_consistency(self, values: List[Dict[str, Any]]) -> float:
        """Calculate consistency between values"""
        if len(values) <= 1:
            return 0.5
        
        # Compare values from different sources
        consistent_pairs = 0
        total_pairs = 0
        
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                total_pairs += 1
                
                # Compare values (simplified)
                val1 = str(values[i]["value"])
                val2 = str(values[j]["value"])
                
                # Simple string similarity
                if val1 == val2:
                    consistent_pairs += 1
                elif self._calculate_string_similarity(val1, val2) > 0.8:
                    consistent_pairs += 1
        
        if total_pairs == 0:
            return 0.5
        
        consistency_score = consistent_pairs / total_pairs
        
        # Weight by source confidences
        avg_confidence = np.mean([v.get("confidence", 0.5) for v in values])
        weighted_score = consistency_score * avg_confidence
        
        return weighted_score
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        # Simple Jaccard similarity
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        
        if not set1 and not set2:
            return 1.0
        elif not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level from score"""
        if score >= self.high_confidence_threshold:
            return "HIGH"
        elif score >= self.medium_confidence_threshold:
            return "MEDIUM"
        elif score >= self.low_confidence_threshold:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def generate_confidence_report(self, 
                                 document_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive confidence report
        
        Args:
            document_results: Complete document processing results
            
        Returns:
            Confidence report
        """
        try:
            report = {
                "summary": {
                    "total_fields": 0,
                    "high_confidence_fields": 0,
                    "medium_confidence_fields": 0,
                    "low_confidence_fields": 0,
                    "average_confidence": 0.0
                },
                "field_analysis": [],
                "recommendations": []
            }
            
            # Analyze each field
            fields = document_results.get("extracted_data", {}).get("fields", {})
            report["summary"]["total_fields"] = len(fields)
            
            confidences = []
            for field_name, field_data in fields.items():
                field_confidence = field_data.get("confidence", 0.0)
                confidences.append(field_confidence)
                
                # Categorize
                confidence_level = self._get_confidence_level(field_confidence)
                
                if confidence_level == "HIGH":
                    report["summary"]["high_confidence_fields"] += 1
                elif confidence_level == "MEDIUM":
                    report["summary"]["medium_confidence_fields"] += 1
                else:
                    report["summary"]["low_confidence_fields"] += 1
                
                # Add field analysis
                report["field_analysis"].append({
                    "field": field_name,
                    "confidence": field_confidence,
                    "level": confidence_level,
                    "value": field_data.get("value", ""),
                    "source": field_data.get("source", "")
                })
            
            # Calculate average
            if confidences:
                report["summary"]["average_confidence"] = np.mean(confidences)
            
            # Generate recommendations
            report["recommendations"] = self._generate_confidence_recommendations(
                report["summary"]
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Confidence report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_confidence_recommendations(self, 
                                           summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on confidence summary"""
        recommendations = []
        
        avg_confidence = summary.get("average_confidence", 0.0)
        low_confidence_fields = summary.get("low_confidence_fields", 0)
        
        if avg_confidence < 0.5:
            recommendations.append(
                "Overall confidence is low. Consider manual review of entire document."
            )
        elif avg_confidence < 0.7:
            recommendations.append(
                "Moderate overall confidence. Review low-confidence fields manually."
            )
        
        if low_confidence_fields > 0:
            recommendations.append(
                f"{low_confidence_fields} fields have low confidence and should be reviewed."
            )
        
        if summary.get("high_confidence_fields", 0) / max(1, summary["total_fields"]) > 0.8:
            recommendations.append(
                "High confidence across most fields. Document is likely reliable."
            )
        
        return recommendations