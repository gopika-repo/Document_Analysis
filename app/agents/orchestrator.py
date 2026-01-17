from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AgentOrchestrator:
    """Main orchestrator for the multi-agent system"""
    
    def __init__(self):
        # Initialize all agents with fallbacks
        self.agents = self._initialize_agents()
        
        # Create workflow graph
        self.workflow = self._create_workflow()
        self.checkpointer = MemorySaver()
        self.compiled_workflow = self.workflow.compile(checkpointer=self.checkpointer)
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents with graceful fallbacks"""
        agents = {}
        
        # Try to import and initialize each agent
        agent_configs = [
            ("quality", "document_quality_agent", "DocumentQualityAgent"),
            ("classifier", "document_type_classifier", "DocumentTypeClassifier"),
            ("layout", "layout_strategy_agent", "LayoutStrategyAgent"),
            ("vision", "visual_element_detector", "VisualElementDetector"),
            ("charts", "chart_understanding_agent", "ChartUnderstandingAgent"),
            ("tables", "table_structure_agent", "TableStructureAgent"),
            ("signatures", "signature_verification_agent", "SignatureVerificationAgent"),
            ("ocr_reliability", "ocr_reliability_agent", "OCRReliabilityAgent"),
            ("entities", "entity_intelligence_agent", "EntityIntelligenceAgent"),
            ("semantics", "semantic_reasoning_agent", "SemanticReasoningAgent"),
            ("alignment", "cross_modal_alignment_agent", "CrossModalAlignmentAgent"),
            ("confidence", "confidence_arbitration_agent", "ConfidenceArbitrationAgent"),
            ("consistency", "temporal_numeric_consistency_agent", "TemporalNumericConsistencyAgent"),
            ("contradiction", "contradiction_detection_agent", "ContradictionDetectionAgent"),
            ("risk", "risk_compliance_agent", "RiskComplianceAgent"),
            ("explanation", "explanation_agent", "ExplanationAgent"),
            ("review", "human_review_agent", "HumanReviewAgent"),
        ]
        
        for key, module_name, agent_class in agent_configs:
            try:
                # Dynamically import the module and agent
                module_path = f"app.agents.{module_name.rsplit('_', 1)[0] if '_agent' in module_name else module_name}.{module_name}"
                module = __import__(module_path, fromlist=[agent_class])
                agent_obj = getattr(module, agent_class)
                agents[key] = agent_obj()
                logger.info(f"Successfully loaded agent: {agent_class}")
            except (ImportError, AttributeError) as e:
                # Create a stub agent if import fails
                agents[key] = self._create_stub_agent(key)
                logger.warning(f"Created stub agent for {key}: {str(e)[:100]}")
        
        return agents
    
    def _create_stub_agent(self, agent_name: str):
        """Create a stub agent that does minimal work"""
        class StubAgent:
            def __init__(self, name):
                self.name = name
            
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.debug(f"Stub agent '{self.name}' executed")
                # Add minimal processing based on agent type
                if self.name == "quality":
                    if not hasattr(state, 'quality_scores'):
                        from app.core.models import QualityScore
                        state.quality_scores = {0: QualityScore(
                            sharpness=0.8, brightness=0.7, contrast=0.6, 
                            noise_level=0.9, overall=0.75
                        )}
                elif self.name == "classifier":
                    from app.core.models import DocumentType
                    state.document_type = DocumentType.UNKNOWN
                elif self.name == "entities":
                    state.extracted_entities = {"dates": [], "amounts": [], "names": []}
                
                return state
        
        return StubAgent(agent_name)
    
    def _create_workflow(self) -> StateGraph:
        """Create the complete agent workflow"""
        workflow = StateGraph(ProcessingState)
        
        # Add all available agents as nodes
        agent_nodes = {
            "assess_quality": "quality",
            "classify_document": "classifier",
            "determine_layout": "layout",
            "detect_elements": "vision",
            "analyze_charts": "charts",
            "analyze_tables": "tables",
            "verify_signatures": "signatures",
            "assess_ocr_reliability": "ocr_reliability",
            "extract_entities": "entities",
            "analyze_semantics": "semantics",
            "align_modalities": "alignment",
            "arbitrate_confidence": "confidence",
            "check_consistency": "consistency",
            "detect_contradictions": "contradiction",
            "assess_risk": "risk",
            "generate_explanations": "explanation",
            "generate_review_recommendations": "review",
        }
        
        for node_name, agent_key in agent_nodes.items():
            if agent_key in self.agents:
                workflow.add_node(node_name, self.agents[agent_key])
        
        # Add compile results node
        workflow.add_node("compile_results", self._compile_final_results)
        
        # Define workflow edges (simplified for now)
        edges = [
            ("assess_quality", "classify_document"),
            ("classify_document", "determine_layout"),
            ("determine_layout", "detect_elements"),
            ("detect_elements", "analyze_charts"),
            ("detect_elements", "analyze_tables"),
            ("detect_elements", "verify_signatures"),
            ("determine_layout", "assess_ocr_reliability"),
            ("assess_ocr_reliability", "extract_entities"),
            ("extract_entities", "analyze_semantics"),
            ("analyze_charts", "align_modalities"),
            ("analyze_tables", "align_modalities"),
            ("verify_signatures", "align_modalities"),
            ("analyze_semantics", "align_modalities"),
            ("align_modalities", "arbitrate_confidence"),
            ("arbitrate_confidence", "check_consistency"),
            ("check_consistency", "detect_contradictions"),
            ("detect_contradictions", "assess_risk"),
            ("assess_risk", "generate_explanations"),
            ("generate_explanations", "generate_review_recommendations"),
            ("generate_review_recommendations", "compile_results"),
        ]
        
        # Add edges only if both nodes exist
        for from_node, to_node in edges:
            if workflow.has_node(from_node) and workflow.has_node(to_node):
                workflow.add_edge(from_node, to_node)
        
        workflow.add_edge("compile_results", END)
        
        # Set entry point
        workflow.set_entry_point("assess_quality")
        
        return workflow
    
    def _compile_final_results(self, state: ProcessingState) -> ProcessingState:
        """Compile final processing results"""
        from datetime import datetime
        from app.core.models import ExtractedField
        
        try:
            logger.info(f"Compiling final results for {state.document_id}")
            
            # Ensure required attributes exist
            if not hasattr(state, 'extracted_entities'):
                state.extracted_entities = {}
            if not hasattr(state, 'chart_analysis'):
                state.chart_analysis = {}
            if not hasattr(state, 'semantic_analysis'):
                state.semantic_analysis = {}
            if not hasattr(state, 'contradictions'):
                state.contradictions = []
            if not hasattr(state, 'errors'):
                state.errors = []
            
            # Calculate integrity score
            integrity_score = self._calculate_integrity_score(state)
            
            # Compile extracted fields
            extracted_fields = {}
            
            # Add entities
            for entity_type, entities in state.extracted_entities.items():
                if entities:
                    extracted_fields[f"entity_{entity_type}"] = ExtractedField(
                        value=entities,
                        confidence=0.7,
                        sources=["text_analysis"],
                        modalities=["textual"]
                    )
            
            # Add chart insights
            for chart_id, analysis in state.chart_analysis.items():
                if isinstance(analysis, dict) and analysis.get("trend_direction") != "unknown":
                    extracted_fields[f"chart_{chart_id}_trend"] = ExtractedField(
                        value=analysis["trend_direction"],
                        confidence=analysis.get("confidence", 0.5),
                        sources=["visual_analysis"],
                        modalities=["visual"]
                    )
            
            # Add semantic insights
            if state.semantic_analysis:
                extracted_fields["semantic_summary"] = ExtractedField(
                    value=state.semantic_analysis.get("summary", "No summary available"),
                    confidence=state.semantic_analysis.get("confidence", 0.6),
                    sources=["semantic_analysis"],
                    modalities=["textual"]
                )
            
            # Prepare final output
            state.extracted_fields = extracted_fields
            state.processing_end = datetime.now()
            
            processing_time = (state.processing_end - state.processing_start).total_seconds()
            
            state.processing_metadata = {
                "integrity_score": integrity_score,
                "total_pages": len(state.images) if hasattr(state, 'images') else 0,
                "agents_executed": list(self.agents.keys()),
                "processing_time": processing_time,
                "document_type": state.document_type.value if hasattr(state, 'document_type') and state.document_type else "unknown"
            }
            
            logger.info(f"Results compiled for {state.document_id}")
            return state
            
        except Exception as e:
            logger.error(f"Results compilation failed: {e}")
            if not hasattr(state, 'errors'):
                state.errors = []
            state.errors.append(f"Results compilation error: {str(e)}")
            return state
    
    def _calculate_integrity_score(self, state: ProcessingState) -> float:
        """Calculate document integrity score"""
        scores = []
        
        # Quality scores (20%)
        if hasattr(state, 'quality_scores') and state.quality_scores:
            avg_quality = sum(score.overall for score in state.quality_scores.values()) / len(state.quality_scores)
            scores.append(avg_quality * 0.2)
        else:
            scores.append(0.6 * 0.2)  # Default score
        
        # OCR confidence (30%)
        if hasattr(state, 'ocr_confidence') and state.ocr_confidence:
            avg_ocr = sum(state.ocr_confidence.values()) / len(state.ocr_confidence)
            scores.append(avg_ocr * 0.3)
        else:
            scores.append(0.7 * 0.3)  # Default score
        
        # Field confidence (30%)
        if hasattr(state, 'field_confidences') and state.field_confidences:
            avg_field = sum(state.field_confidences.values()) / len(state.field_confidences)
            scores.append(avg_field * 0.3)
        else:
            scores.append(0.65 * 0.3)  # Default score
        
        # Contradiction penalty (20%)
        contradiction_count = len(state.contradictions) if hasattr(state, 'contradictions') else 0
        contradiction_penalty = contradiction_count * 0.1
        scores.append(max(0, 0.2 - contradiction_penalty))
        
        return min(1.0, sum(scores))
    
    async def process_document(self, images: List, file_path: str = None) -> Dict[str, Any]:
        """Main method to process a document"""
        try:
            logger.info("Starting document processing pipeline")
            
            # Initialize state
            state = ProcessingState(
                file_path=file_path,
                images=images
            )
            
            # Execute workflow
            final_state = await self.compiled_workflow.ainvoke(
                state,
                config={"configurable": {"thread_id": state.document_id}}
            )
            
            # Prepare response
            response = {
                "success": len(final_state.errors) == 0,
                "document_id": final_state.document_id,
                "document_type": final_state.document_type.value if hasattr(final_state, 'document_type') and final_state.document_type else "unknown",
                "extracted_fields": {
                    name: {
                        "value": field.value,
                        "confidence": field.confidence,
                        "sources": field.sources,
                        "modalities": field.modalities
                    }
                    for name, field in final_state.extracted_fields.items()
                } if hasattr(final_state, 'extracted_fields') else {},
                "validation_results": {
                    "contradictions": [
                        {
                            "type": c.contradiction_type.value if hasattr(c.contradiction_type, 'value') else str(c.contradiction_type),
                            "severity": c.severity.value if hasattr(c.severity, 'value') else str(c.severity),
                            "explanation": c.explanation,
                            "confidence": c.confidence
                        }
                        for c in final_state.contradictions
                    ] if hasattr(final_state, 'contradictions') else [],
                    "risk_score": final_state.risk_score if hasattr(final_state, 'risk_score') else 0.0,
                    "integrity_score": final_state.processing_metadata.get("integrity_score", 0.0) if hasattr(final_state, 'processing_metadata') else 0.0
                },
                "explanations": final_state.explanations if hasattr(final_state, 'explanations') else {},
                "recommendations": final_state.review_recommendations if hasattr(final_state, 'review_recommendations') else [],
                "processing_metadata": final_state.processing_metadata if hasattr(final_state, 'processing_metadata') else {},
                "errors": final_state.errors if hasattr(final_state, 'errors') else []
            }
            
            logger.info(f"Document processing completed: {final_state.document_id}")
            return response
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_id": state.document_id if 'state' in locals() else "unknown"
            }