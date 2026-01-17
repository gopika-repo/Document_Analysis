from typing import Dict, List, Any
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ConfidenceArbitrationAgent:
    """Agent for arbitrating confidence scores across different modalities"""
    
    def __init__(self):
        self.modality_weights = {
            "visual": 0.4,
            "textual": 0.4,
            "semantic": 0.2
        }
        
        self.source_weights = {
            "yolo_detection": 0.8,
            "tesseract_ocr": 0.7,
            "regex_entity": 0.6,
            "heuristic": 0.5,
            "fallback": 0.3
        }
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Arbitrate confidence scores across modalities"""
        try:
            logger.info(f"Arbitrating confidence scores for {state.document_id}")
            
            field_confidences = {}
            
            # Calculate confidence for different field types
            field_confidences["document_quality"] = self._calculate_quality_confidence(state)
            field_confidences["ocr_reliability"] = self._calculate_ocr_confidence(state)
            field_confidences["entity_extraction"] = self._calculate_entity_confidence(state)
            field_confidences["visual_analysis"] = self._calculate_visual_confidence(state)
            field_confidences["semantic_understanding"] = self._calculate_semantic_confidence(state)
            field_confidences["cross_modal_alignment"] = self._calculate_alignment_confidence(state)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(field_confidences)
            field_confidences["overall"] = overall_confidence
            
            state.field_confidences = field_confidences
            logger.info(f"Confidence arbitration completed: overall {overall_confidence:.2f}")
            
            return state
            
        except Exception as e:
            logger.error(f"Confidence arbitration failed: {e}")
            state.field_confidences = {"error": str(e), "overall": 0.3}
            return state
    
    def _calculate_quality_confidence(self, state: ProcessingState) -> float:
        """Calculate confidence based on document quality"""
        try:
            if hasattr(state, 'quality_scores') and state.quality_scores:
                # Average quality across pages
                quality_scores = [score.overall for score in state.quality_scores.values()]
                return sum(quality_scores) / len(quality_scores)
        except:
            pass
        
        return 0.5  # Default
    
    def _calculate_ocr_confidence(self, state: ProcessingState) -> float:
        """Calculate confidence in OCR results"""
        try:
            if hasattr(state, 'ocr_confidence') and state.ocr_confidence:
                # Average OCR confidence across pages
                return sum(state.ocr_confidence.values()) / len(state.ocr_confidence)
        except:
            pass
        
        return 0.6  # Default
    
    def _calculate_entity_confidence(self, state: ProcessingState) -> float:
        """Calculate confidence in entity extraction"""
        try:
            if hasattr(state, 'extracted_entities') and state.extracted_entities:
                # Calculate based on number and diversity of entities
                total_entities = sum(len(entities) for entities in state.extracted_entities.values())
                unique_entity_types = len(state.extracted_entities)
                
                entity_richness = min(total_entities / 20, 1.0)  # Cap at 20 entities
                type_diversity = min(unique_entity_types / 5, 1.0)  # Cap at 5 types
                
                return (entity_richness * 0.6 + type_diversity * 0.4)
        except:
            pass
        
        return 0.4  # Default
    
    def _calculate_visual_confidence(self, state: ProcessingState) -> float:
        """Calculate confidence in visual analysis"""
        try:
            if hasattr(state, 'visual_elements') and state.visual_elements:
                total_elements = sum(len(elements) for elements in state.visual_elements.values())
                
                if total_elements > 0:
                    # Calculate average element confidence
                    all_confidences = []
                    for elements in state.visual_elements.values():
                        for element in elements:
                            if hasattr(element, 'confidence'):
                                all_confidences.append(element.confidence)
                    
                    if all_confidences:
                        avg_confidence = sum(all_confidences) / len(all_confidences)
                        
                        # Factor in number of elements
                        element_factor = min(total_elements / 10, 1.0)
                        
                        return avg_confidence * 0.7 + element_factor * 0.3
        except:
            pass
        
        return 0.5  # Default
    
    def _calculate_semantic_confidence(self, state: ProcessingState) -> float:
        """Calculate confidence in semantic understanding"""
        try:
            if hasattr(state, 'semantic_analysis') and state.semantic_analysis:
                return state.semantic_analysis.get('confidence', 0.5)
        except:
            pass
        
        return 0.5  # Default
    
    def _calculate_alignment_confidence(self, state: ProcessingState) -> float:
        """Calculate confidence in cross-modal alignment"""
        try:
            if hasattr(state, 'aligned_data') and state.aligned_data:
                return state.aligned_data.get('alignment_confidence', 0.5)
        except:
            pass
        
        return 0.4  # Default
    
    def _calculate_overall_confidence(self, field_confidences: Dict[str, float]) -> float:
        """Calculate overall confidence score"""
        # Weight different confidence types
        weights = {
            "document_quality": 0.15,
            "ocr_reliability": 0.20,
            "entity_extraction": 0.20,
            "visual_analysis": 0.20,
            "semantic_understanding": 0.15,
            "cross_modal_alignment": 0.10
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for field, weight in weights.items():
            if field in field_confidences:
                weighted_sum += field_confidences[field] * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        
        return 0.5  # Default overall confidence
    
    def _arbitrate_specific_field(self, field_name: str, 
                                modality_scores: Dict[str, float],
                                source_scores: Dict[str, float]) -> Dict[str, Any]:
        """Arbitrate confidence for a specific field"""
        arbitration = {
            "field": field_name,
            "modality_scores": modality_scores,
            "source_scores": source_scores,
            "final_confidence": 0.0,
            "dominant_modality": None,
            "dominant_source": None
        }
        
        # Calculate weighted modality score
        modality_total = 0
        modality_weighted = 0
        
        for modality, score in modality_scores.items():
            weight = self.modality_weights.get(modality, 0.3)
            modality_weighted += score * weight
            modality_total += weight
        
        modality_score = modality_weighted / modality_total if modality_total > 0 else 0.5
        
        # Calculate weighted source score
        source_total = 0
        source_weighted = 0
        
        for source, score in source_scores.items():
            weight = self.source_weights.get(source, 0.5)
            source_weighted += score * weight
            source_total += weight
        
        source_score = source_weighted / source_total if source_total > 0 else 0.5
        
        # Final confidence is average of modality and source scores
        arbitration["final_confidence"] = (modality_score + source_score) / 2
        
        # Determine dominant modality and source
        if modality_scores:
            arbitration["dominant_modality"] = max(modality_scores.items(), key=lambda x: x[1])[0]
        
        if source_scores:
            arbitration["dominant_source"] = max(source_scores.items(), key=lambda x: x[1])[0]
        
        return arbitration