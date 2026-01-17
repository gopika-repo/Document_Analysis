from typing import Dict, List, Any
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class TableStructureAgent:
    """Agent for analyzing table structures in documents"""
    
    def __init__(self):
        self.min_cell_area = 100  # Minimum area for a cell
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Analyze table structures in detected table elements"""
        try:
            logger.info(f"Analyzing table structures for {state.document_id}")
            
            table_structures = {}
            
            if hasattr(state, 'visual_elements') and state.visual_elements:
                # Find table elements
                tables = self._extract_table_elements(state.visual_elements)
                
                for table_id, (page_num, table_element) in enumerate(tables):
                    structure = self._analyze_table_structure(table_element, state.images[page_num] if page_num < len(state.images) else None)
                    table_structures[f"table_{table_id}"] = structure
            
            state.table_structures = table_structures
            logger.info(f"Analyzed {len(table_structures)} tables")
            
            return state
            
        except Exception as e:
            logger.error(f"Table structure analysis failed: {e}")
            state.table_structures = {}
            return state
    
    def _extract_table_elements(self, visual_elements: Dict[int, List[Any]]) -> List[tuple]:
        """Extract table elements from visual detections"""
        tables = []
        
        for page_num, elements in visual_elements.items():
            for element in elements:
                if hasattr(element, 'element_type') and element.element_type == 'table':
                    tables.append((page_num, element))
        
        return tables
    
    def _analyze_table_structure(self, table_element, image) -> Dict[str, Any]:
        """Analyze the structure of a table"""
        structure = {
            "page": table_element.page_num if hasattr(table_element, 'page_num') else 0,
            "bbox": table_element.bbox if hasattr(table_element, 'bbox') else [],
            "confidence": table_element.confidence if hasattr(table_element, 'confidence') else 0.0,
            "rows": 0,
            "columns": 0,
            "has_header": False,
            "has_totals": False,
            "cell_count": 0,
            "structure_type": "unknown"
        }
        
        try:
            # Simple heuristic analysis
            bbox = structure["bbox"]
            if len(bbox) == 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Estimate rows and columns based on aspect ratio
                if width > height:  # Wide table
                    structure["rows"] = 3
                    structure["columns"] = 5
                    structure["structure_type"] = "wide"
                else:  # Tall table
                    structure["rows"] = 5
                    structure["columns"] = 3
                    structure["structure_type"] = "tall"
                
                structure["cell_count"] = structure["rows"] * structure["columns"]
                structure["has_header"] = True  # Assume tables have headers
                
                # Check if table might have totals (usually last row)
                if structure["rows"] > 3:
                    structure["has_totals"] = True
        
        except Exception as e:
            logger.warning(f"Table structure analysis failed: {e}")
        
        return structure