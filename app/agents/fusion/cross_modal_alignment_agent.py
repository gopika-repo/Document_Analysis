from typing import Dict, List, Any, Tuple
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class CrossModalAlignmentAgent:
    """Agent for aligning data across different modalities (text, vision, etc.)"""
    
    def __init__(self):
        self.alignment_threshold = 0.6
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Align data from different modalities"""
        try:
            logger.info(f"Aligning multi-modal data for {state.document_id}")
            
            aligned_data = {
                "text_visual_alignments": [],
                "entity_region_mappings": [],
                "cross_references": [],
                "alignment_confidence": 0.0
            }
            
            # Align text entities with visual regions
            text_visual_alignments = self._align_text_with_visual(state)
            aligned_data["text_visual_alignments"] = text_visual_alignments
            
            # Map entities to document regions
            entity_mappings = self._map_entities_to_regions(state)
            aligned_data["entity_region_mappings"] = entity_mappings
            
            # Find cross-references between different parts
            cross_refs = self._find_cross_references(state)
            aligned_data["cross_references"] = cross_refs
            
            # Calculate overall alignment confidence
            alignment_confidence = self._calculate_alignment_confidence(
                text_visual_alignments, entity_mappings, cross_refs
            )
            aligned_data["alignment_confidence"] = alignment_confidence
            
            state.aligned_data = aligned_data
            logger.info(f"Cross-modal alignment completed: confidence {alignment_confidence:.2f}")
            
            return state
            
        except Exception as e:
            logger.error(f"Cross-modal alignment failed: {e}")
            state.aligned_data = {"error": str(e)}
            return state
    
    def _align_text_with_visual(self, state: ProcessingState) -> List[Dict[str, Any]]:
        """Align text content with visual elements"""
        alignments = []
        
        try:
            # Check if we have both text and visual elements
            if not hasattr(state, 'ocr_results') or not hasattr(state, 'visual_elements'):
                return alignments
            
            # Simple spatial alignment based on bounding boxes
            for page_num, ocr_data in state.ocr_results.items():
                if page_num in state.visual_elements:
                    page_alignments = self._align_page_elements(
                        ocr_data, state.visual_elements[page_num], page_num
                    )
                    alignments.extend(page_alignments)
        
        except Exception as e:
            logger.warning(f"Text-visual alignment failed: {e}")
        
        return alignments
    
    def _align_page_elements(self, ocr_data: Any, visual_elements: List, page_num: int) -> List[Dict[str, Any]]:
        """Align elements on a single page"""
        alignments = []
        
        try:
            # Extract text blocks from OCR
            text_blocks = self._extract_text_blocks(ocr_data)
            
            for visual_element in visual_elements:
                if hasattr(visual_element, 'bbox'):
                    # Find text blocks near this visual element
                    nearby_text = self._find_nearby_text(text_blocks, visual_element.bbox)
                    
                    if nearby_text:
                        alignment = {
                            "page": page_num,
                            "visual_element": {
                                "type": visual_element.element_type if hasattr(visual_element, 'element_type') else "unknown",
                                "bbox": visual_element.bbox,
                                "confidence": visual_element.confidence if hasattr(visual_element, 'confidence') else 0.0
                            },
                            "text_context": nearby_text[:200],  # First 200 chars
                            "alignment_score": self._calculate_spatial_alignment(
                                visual_element.bbox, nearby_text
                            ),
                            "relationship": self._infer_relationship(visual_element, nearby_text)
                        }
                        alignments.append(alignment)
        
        except Exception as e:
            logger.warning(f"Page alignment failed: {e}")
        
        return alignments
    
    def _extract_text_blocks(self, ocr_data: Any) -> List[Dict[str, Any]]:
        """Extract text blocks from OCR data"""
        text_blocks = []
        
        try:
            if isinstance(ocr_data, dict):
                # Try to extract structured text blocks
                if 'words' in ocr_data and isinstance(ocr_data['words'], list):
                    for word in ocr_data['words']:
                        if isinstance(word, dict) and 'bbox' in word and 'text' in word:
                            text_blocks.append({
                                "text": word['text'],
                                "bbox": word['bbox'],
                                "confidence": word.get('confidence', 0.0)
                            })
                elif 'text' in ocr_data:
                    # Fallback: treat entire page as one block
                    text_blocks.append({
                        "text": ocr_data['text'],
                        "bbox": [0, 0, 1000, 1000],  # Default bbox
                        "confidence": ocr_data.get('average_confidence', 0.5)
                    })
            elif isinstance(ocr_data, str):
                text_blocks.append({
                    "text": ocr_data,
                    "bbox": [0, 0, 1000, 1000],
                    "confidence": 0.5
                })
        
        except Exception as e:
            logger.warning(f"Text block extraction failed: {e}")
        
        return text_blocks
    
    def _find_nearby_text(self, text_blocks: List[Dict[str, Any]], visual_bbox: List[int]) -> str:
        """Find text near a visual element"""
        nearby_text = []
        
        try:
            if len(visual_bbox) != 4:
                return ""
            
            vx1, vy1, vx2, vy2 = visual_bbox
            visual_center_x = (vx1 + vx2) / 2
            visual_center_y = (vy1 + vy2) / 2
            
            for block in text_blocks:
                if 'bbox' in block and len(block['bbox']) == 4:
                    bx1, by1, bx2, by2 = block['bbox']
                    block_center_x = (bx1 + bx2) / 2
                    block_center_y = (by1 + by2) / 2
                    
                    # Calculate distance between centers
                    distance = ((block_center_x - visual_center_x) ** 2 + 
                               (block_center_y - visual_center_y) ** 2) ** 0.5
                    
                    # If within reasonable distance (500 pixels)
                    if distance < 500:
                        nearby_text.append(block.get('text', ''))
        
        except Exception as e:
            logger.warning(f"Nearby text finding failed: {e}")
        
        return " ".join(nearby_text)
    
    def _calculate_spatial_alignment(self, visual_bbox: List[int], text: str) -> float:
        """Calculate spatial alignment score"""
        # Simple heuristic: longer nearby text suggests better alignment
        text_length = len(text)
        
        if text_length > 100:
            return 0.8
        elif text_length > 50:
            return 0.6
        elif text_length > 20:
            return 0.4
        else:
            return 0.2
    
    def _infer_relationship(self, visual_element, text: str) -> str:
        """Infer relationship between visual element and text"""
        element_type = visual_element.element_type if hasattr(visual_element, 'element_type') else "unknown"
        text_lower = text.lower()
        
        relationships = {
            "table": ["shows", "presents", "displays", "illustrates"],
            "chart": ["shows", "illustrates", "depicts", "represents"],
            "figure": ["shows", "illustrates", "depicts"],
            "signature": ["signed", "authorized", "approved", "witnessed"]
        }
        
        if element_type in relationships:
            for keyword in relationships[element_type]:
                if keyword in text_lower:
                    return f"text_{keyword}_{element_type}"
        
        return f"text_near_{element_type}"
    
    def _map_entities_to_regions(self, state: ProcessingState) -> List[Dict[str, Any]]:
        """Map extracted entities to document regions"""
        mappings = []
        
        try:
            if not hasattr(state, 'extracted_entities') or not state.extracted_entities:
                return mappings
            
            # Simple mapping: each entity type gets a region estimate
            for entity_type, entities in state.extracted_entities.items():
                if entities:
                    mapping = {
                        "entity_type": entity_type,
                        "entities_count": len(entities),
                        "sample_entities": entities[:3],  # First 3 as sample
                        "estimated_region": self._estimate_entity_region(entity_type, entities),
                        "confidence": min(len(entities) / 10, 1.0)  # More entities = higher confidence
                    }
                    mappings.append(mapping)
        
        except Exception as e:
            logger.warning(f"Entity region mapping failed: {e}")
        
        return mappings
    
    def _estimate_entity_region(self, entity_type: str, entities: List[str]) -> str:
        """Estimate where entities are typically found in documents"""
        region_map = {
            "dates": "header/footer",
            "amounts": "body/tables",
            "percentages": "body/charts",
            "phone_numbers": "contact_info",
            "emails": "contact_info",
            "companies": "header",
            "names": "header/signature"
        }
        
        return region_map.get(entity_type, "body")
    
    def _find_cross_references(self, state: ProcessingState) -> List[Dict[str, Any]]:
        """Find cross-references between different document parts"""
        cross_refs = []
        
        try:
            # Look for references between text and visual elements
            if hasattr(state, 'semantic_analysis') and state.semantic_analysis:
                summary = state.semantic_analysis.get('summary', '')
                
                # Check for references to figures, tables, etc.
                reference_patterns = [
                    (r'figure\s+(\d+)', 'figure'),
                    (r'table\s+(\d+)', 'table'),
                    (r'chart\s+(\d+)', 'chart'),
                    (r'graph\s+(\d+)', 'chart')
                ]
                
                for pattern, ref_type in reference_patterns:
                    import re
                    matches = re.findall(pattern, summary, re.IGNORECASE)
                    for match in matches:
                        cross_refs.append({
                            "type": f"text_references_{ref_type}",
                            "reference": f"{ref_type} {match}",
                            "context": summary[:100],
                            "confidence": 0.7
                        })
        
        except Exception as e:
            logger.warning(f"Cross-reference finding failed: {e}")
        
        return cross_refs
    
    def _calculate_alignment_confidence(self, text_alignments: List, 
                                      entity_mappings: List, 
                                      cross_refs: List) -> float:
        """Calculate overall alignment confidence"""
        factors = []
        
        # Text-visual alignment factor
        if text_alignments:
            avg_alignment_score = sum(a.get('alignment_score', 0) for a in text_alignments) / len(text_alignments)
            factors.append(avg_alignment_score * 0.4)
        
        # Entity mapping factor
        if entity_mappings:
            avg_mapping_confidence = sum(m.get('confidence', 0) for m in entity_mappings) / len(entity_mappings)
            factors.append(avg_mapping_confidence * 0.3)
        
        # Cross-reference factor
        if cross_refs:
            factors.append(0.2)
        
        # Base confidence if no alignments
        if not factors:
            return 0.3
        
        return min(sum(factors), 1.0)