from typing import Dict, Any
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class LayoutStrategyAgent:
    """Agent for determining the optimal processing strategy based on document layout"""
    
    def __init__(self):
        self.strategies = {
            "text_heavy": {"vision_weight": 0.2, "text_weight": 0.8, "ocr_first": True},
            "visual_heavy": {"vision_weight": 0.7, "text_weight": 0.3, "ocr_first": False},
            "balanced": {"vision_weight": 0.5, "text_weight": 0.5, "ocr_first": True},
            "tabular": {"vision_weight": 0.6, "text_weight": 0.4, "table_first": True},
        }
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Determine processing strategy based on document analysis"""
        try:
            logger.info(f"Determining layout strategy for {state.document_id}")
            
            # Analyze document characteristics
            characteristics = self._analyze_document_characteristics(state)
            
            # Determine optimal strategy
            strategy = self._determine_strategy(characteristics)
            
            # Apply strategy to state
            state.layout_strategy = strategy
            state.processing_metadata = state.processing_metadata or {}
            state.processing_metadata["layout_strategy"] = strategy
            state.processing_metadata["strategy_weights"] = self.strategies.get(strategy, {})
            
            logger.info(f"Selected strategy: {strategy}")
            return state
            
        except Exception as e:
            logger.error(f"Layout strategy determination failed: {e}")
            state.layout_strategy = "balanced"  # Default fallback
            return state
    
    def _analyze_document_characteristics(self, state: ProcessingState) -> Dict[str, float]:
        """Analyze document to determine its characteristics"""
        characteristics = {
            "text_density": 0.5,
            "visual_complexity": 0.5,
            "table_presence": 0.0,
            "chart_presence": 0.0,
        }
        
        # Use OCR results if available
        if hasattr(state, 'ocr_results') and state.ocr_results:
            text_density = self._calculate_text_density(state.ocr_results)
            characteristics["text_density"] = text_density
        
        # Use visual elements if available
        if hasattr(state, 'visual_elements') and state.visual_elements:
            visual_analysis = self._analyze_visual_elements(state.visual_elements)
            characteristics.update(visual_analysis)
        
        return characteristics
    
    def _calculate_text_density(self, ocr_results) -> float:
        """Calculate text density from OCR results"""
        try:
            total_text = ""
            for result in ocr_results.values():
                if isinstance(result, dict) and 'text' in result:
                    total_text += result['text']
            
            # Simple heuristic: longer text = higher density
            text_length = len(total_text)
            if text_length > 1000:
                return 0.8
            elif text_length > 500:
                return 0.6
            elif text_length > 100:
                return 0.4
            else:
                return 0.2
        except:
            return 0.5
    
    def _analyze_visual_elements(self, visual_elements) -> Dict[str, float]:
        """Analyze visual elements for layout complexity"""
        analysis = {
            "visual_complexity": 0.0,
            "table_presence": 0.0,
            "chart_presence": 0.0,
        }
        
        try:
            total_elements = 0
            table_count = 0
            chart_count = 0
            
            for page_elements in visual_elements.values():
                if isinstance(page_elements, list):
                    total_elements += len(page_elements)
                    for element in page_elements:
                        if hasattr(element, 'element_type'):
                            if element.element_type == 'table':
                                table_count += 1
                            elif element.element_type == 'chart':
                                chart_count += 1
            
            # Calculate scores
            if total_elements > 0:
                analysis["visual_complexity"] = min(total_elements / 10, 1.0)
                analysis["table_presence"] = min(table_count / 3, 1.0)
                analysis["chart_presence"] = min(chart_count / 2, 1.0)
        
        except Exception as e:
            logger.warning(f"Visual analysis failed: {e}")
        
        return analysis
    
    def _determine_strategy(self, characteristics: Dict[str, float]) -> str:
        """Determine the optimal processing strategy"""
        text_density = characteristics.get("text_density", 0.5)
        visual_complexity = characteristics.get("visual_complexity", 0.5)
        table_presence = characteristics.get("table_presence", 0.0)
        chart_presence = characteristics.get("chart_presence", 0.0)
        
        # Decision logic
        if table_presence > 0.7:
            return "tabular"
        elif text_density > 0.7 and visual_complexity < 0.3:
            return "text_heavy"
        elif visual_complexity > 0.7 or (chart_presence > 0.5 and table_presence > 0.5):
            return "visual_heavy"
        else:
            return "balanced"