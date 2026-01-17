from typing import Dict, Any, List, Optional
import numpy as np
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from app.models.yolo_loader import YOLOModel, DetectionResult
from app.models.layout_analyzer import LayoutAnalyzer
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class VisionState(BaseModel):
    """State for Vision Agent"""
    document_id: str
    images: List[np.ndarray] = Field(default_factory=list)
    detections: Dict[int, List[DetectionResult]] = Field(default_factory=dict)
    layout_analysis: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    visual_summary: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

class VisionAgent:
    """Vision Agent for visual analysis of documents"""
    
    def __init__(self, yolo_model: Optional[YOLOModel] = None):
        self.yolo_model = yolo_model or YOLOModel()
        self.layout_analyzer = LayoutAnalyzer()
        
    def create_graph(self) -> StateGraph:
        """Create LangGraph for vision processing"""
        workflow = StateGraph(VisionState)
        
        # Add nodes
        workflow.add_node("detect_elements", self.detect_elements)
        workflow.add_node("analyze_layout", self.analyze_layout)
        workflow.add_node("generate_summary", self.generate_summary)
        
        # Add edges
        workflow.add_edge("detect_elements", "analyze_layout")
        workflow.add_edge("analyze_layout", "generate_summary")
        workflow.add_edge("generate_summary", END)
        
        # Set entry point
        workflow.set_entry_point("detect_elements")
        
        return workflow
    
    def detect_elements(self, state: VisionState) -> VisionState:
        """Detect visual elements using YOLO"""
        try:
            logger.info(f"Detecting elements for document {state.document_id}")
            
            detections = {}
            for idx, image in enumerate(state.images):
                page_detections = self.yolo_model.detect(image, page_num=idx)
                detections[idx] = page_detections
            
            state.detections = detections
            logger.info(f"Detected elements on {len(detections)} pages")
            
        except Exception as e:
            error_msg = f"Element detection failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def analyze_layout(self, state: VisionState) -> VisionState:
        """Analyze document layout"""
        try:
            logger.info(f"Analyzing layout for document {state.document_id}")
            
            layouts = {}
            for idx, image in enumerate(state.images):
                page_detections = state.detections.get(idx, [])
                layout = self.layout_analyzer.analyze(
                    image, detections=page_detections, page_num=idx
                )
                layouts[idx] = layout
            
            state.layout_analysis = layouts
            logger.info(f"Layout analysis completed for {len(layouts)} pages")
            
        except Exception as e:
            error_msg = f"Layout analysis failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def generate_summary(self, state: VisionState) -> VisionState:
        """Generate visual summary"""
        try:
            logger.info(f"Generating visual summary for document {state.document_id}")
            
            # Count detections by type
            detection_counts = {}
            for page_detections in state.detections.values():
                for detection in page_detections:
                    class_name = detection.class_name
                    detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
            
            # Extract key visual information
            visual_summary = {
                "document_id": state.document_id,
                "total_pages": len(state.images),
                "detection_summary": detection_counts,
                "layout_stats": {
                    "total_regions": sum(
                        len(layout.get("regions", []))
                        for layout in state.layout_analysis.values()
                    ),
                    "pages_with_tables": sum(
                        1 for layout in state.layout_analysis.values()
                        if any(r.get("type") == "table" 
                              for r in layout.get("regions", []))
                    ),
                    "pages_with_charts": sum(
                        1 for layout in state.layout_analysis.values()
                        if any(r.get("type") == "chart" 
                              for r in layout.get("regions", []))
                    )
                },
                "confidence_scores": {
                    "average_detection_confidence": self._calculate_average_confidence(
                        state.detections
                    ),
                    "layout_confidence": 0.85  # Placeholder
                }
            }
            
            state.visual_summary = visual_summary
            logger.info("Visual summary generated successfully")
            
        except Exception as e:
            error_msg = f"Summary generation failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def _calculate_average_confidence(self, 
                                    detections: Dict[int, List[DetectionResult]]) -> float:
        """Calculate average confidence across all detections"""
        if not detections:
            return 0.0
        
        all_confidences = []
        for page_detections in detections.values():
            for detection in page_detections:
                all_confidences.append(detection.confidence)
        
        return np.mean(all_confidences) if all_confidences else 0.0
    
    async def process_document(self, document_id: str, 
                              images: List[np.ndarray]) -> Dict[str, Any]:
        """Process document through vision pipeline"""
        try:
            # Initialize state
            state = VisionState(document_id=document_id, images=images)
            
            # Create and run graph
            graph = self.create_graph()
            compiled_graph = graph.compile()
            
            # Execute graph
            result_state = compiled_graph.invoke(state)
            
            # Prepare response
            response = {
                "success": len(result_state.errors) == 0,
                "document_id": result_state.document_id,
                "visual_summary": result_state.visual_summary,
                "detections": {
                    str(page): [
                        {
                            "class": det.class_name,
                            "confidence": det.confidence,
                            "bbox": det.bbox
                        }
                        for det in detections
                    ]
                    for page, detections in result_state.detections.items()
                },
                "errors": result_state.errors
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Vision processing failed: {e}")
            return {
                "success": False,
                "document_id": document_id,
                "error": str(e)
            }