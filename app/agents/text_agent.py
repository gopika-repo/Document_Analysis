import numpy as np
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from app.models.ocr_engine import HybridOCREngine, OCRResult
from app.config import settings
from app.utils.logger import setup_logger
import requests
import json

logger = setup_logger(__name__)

class TextState(BaseModel):
    """State for Text Agent"""
    document_id: str
    images: List[Any] = Field(default_factory=list)
    ocr_results: List[OCRResult] = Field(default_factory=list)
    text_summary: Dict[str, Any] = Field(default_factory=dict)
    extracted_entities: Dict[str, List[str]] = Field(default_factory=dict)
    semantic_analysis: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

class TextAgent:
    """Text Agent for OCR and semantic analysis"""
    
    def __init__(self):
        self.ocr_engine = HybridOCREngine()
        self.llm_client = self._initialize_llm_client()
        
    def _initialize_llm_client(self):
        """Initialize LLM client based on configuration"""
        if settings.GROK_API_KEY:
            logger.info("Using Grok API for text analysis")
            return GrokClient()
        else:
            logger.info("Using Ollama fallback for text analysis")
            return OllamaClient()
    
    def create_graph(self) -> StateGraph:
        """Create LangGraph for text processing"""
        workflow = StateGraph(TextState)
        
        # Add nodes
        workflow.add_node("perform_ocr", self.perform_ocr)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("analyze_semantics", self.analyze_semantics)
        workflow.add_node("generate_summary", self.generate_text_summary)
        
        # Add edges
        workflow.add_edge("perform_ocr", "extract_entities")
        workflow.add_edge("extract_entities", "analyze_semantics")
        workflow.add_edge("analyze_semantics", "generate_summary")
        workflow.add_edge("generate_summary", END)
        
        # Set entry point
        workflow.set_entry_point("perform_ocr")
        
        return workflow
    
    def perform_ocr(self, state: TextState) -> TextState:
        """Perform OCR on document images"""
        try:
            logger.info(f"Performing OCR for document {state.document_id}")
            
            # Convert images to numpy arrays if needed
            images = []
            for img in state.images:
                if isinstance(img, np.ndarray):
                    images.append(img)
                else:
                    # Convert PIL or other formats
                    images.append(np.array(img))
            
            # Run OCR
            ocr_results = self.ocr_engine.process_batch(images)
            state.ocr_results = ocr_results
            
            total_words = sum(len(result.words) for result in ocr_results)
            avg_confidence = np.mean([r.average_confidence for r in ocr_results])
            
            logger.info(f"OCR completed: {total_words} words, avg confidence: {avg_confidence:.2f}")
            
        except Exception as e:
            error_msg = f"OCR processing failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def extract_entities(self, state: TextState) -> TextState:
        """Extract entities from OCR text"""
        try:
            logger.info(f"Extracting entities for document {state.document_id}")
            
            # Combine all OCR text
            all_text = "\n".join([result.text for result in state.ocr_results])
            
            # Extract entities using LLM
            entities = self._extract_entities_with_llm(all_text)
            state.extracted_entities = entities
            
            logger.info(f"Extracted {sum(len(v) for v in entities.values())} entities")
            
        except Exception as e:
            error_msg = f"Entity extraction failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def _extract_entities_with_llm(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using LLM"""
        prompt = f"""
        Extract key entities from the following document text.
        Return as JSON with categories: dates, amounts, names, organizations, locations, other_fields.
        
        Text:
        {text[:5000]}  # Limit text length
        
        JSON Response:
        """
        
        try:
            response = self.llm_client.generate(prompt)
            
            # Parse response
            if isinstance(response, dict):
                return response
            else:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return self._fallback_entity_extraction(text)
                    
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")
            return self._fallback_entity_extraction(text)
    
    def _fallback_entity_extraction(self, text: str) -> Dict[str, List[str]]:
        """Fallback entity extraction using regex"""
        import re
        
        entities = {
            "dates": re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text),
            "amounts": re.findall(r'\$\d+(?:\.\d{2})?|₹\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP)', text),
            "names": re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text),
            "organizations": re.findall(r'\b(?:Inc|Ltd|LLC|Corp|Company)\b', text, re.IGNORECASE),
            "locations": [],
            "other_fields": []
        }
        
        return {k: list(set(v)) for k, v in entities.items()}
    
    def analyze_semantics(self, state: TextState) -> TextState:
        """Perform semantic analysis"""
        try:
            logger.info(f"Performing semantic analysis for document {state.document_id}")
            
            # Combine all OCR text
            all_text = "\n".join([result.text for result in state.ocr_results])
            
            # Get semantic analysis from LLM
            analysis = self._analyze_semantics_with_llm(all_text)
            state.semantic_analysis = analysis
            
            logger.info("Semantic analysis completed")
            
        except Exception as e:
            error_msg = f"Semantic analysis failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def _analyze_semantics_with_llm(self, text: str) -> Dict[str, Any]:
        """Analyze semantics using LLM"""
        prompt = f"""
        Analyze the following document text and provide:
        1. Document type (invoice, report, form, etc.)
        2. Key topics/themes
        3. Sentiment (positive, negative, neutral)
        4. Writing style (formal, informal, technical)
        5. Key findings/summary
        
        Text:
        {text[:3000]}
        
        Return as JSON.
        """
        
        try:
            response = self.llm_client.generate(prompt)
            
            if isinstance(response, dict):
                return response
            else:
                return {
                    "document_type": "unknown",
                    "topics": [],
                    "sentiment": "neutral",
                    "writing_style": "unknown",
                    "summary": "Analysis unavailable"
                }
                
        except Exception as e:
            logger.warning(f"LLM semantic analysis failed: {e}")
            return {
                "document_type": "unknown",
                "topics": [],
                "sentiment": "neutral",
                "writing_style": "unknown",
                "summary": "Analysis failed"
            }
    
    def generate_text_summary(self, state: TextState) -> TextState:
        """Generate text summary"""
        try:
            logger.info(f"Generating text summary for document {state.document_id}")
            
            # Calculate statistics
            total_words = sum(len(result.words) for result in state.ocr_results)
            avg_confidence = np.mean([r.average_confidence for r in state.ocr_results])
            
            text_summary = {
                "document_id": state.document_id,
                "ocr_statistics": {
                    "total_pages": len(state.ocr_results),
                    "total_words": total_words,
                    "average_confidence": avg_confidence,
                    "ocr_engines_used": list(set(r.engine_used for r in state.ocr_results))
                },
                "entity_summary": {
                    category: len(entities)
                    for category, entities in state.extracted_entities.items()
                },
                "semantic_analysis": state.semantic_analysis,
                "full_text_available": total_words > 0
            }
            
            state.text_summary = text_summary
            logger.info("Text summary generated successfully")
            
        except Exception as e:
            error_msg = f"Summary generation failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def process_document(self, document_id: str, 
                              images: List[Any]) -> Dict[str, Any]:
        """Process document through text pipeline"""
        try:
            # Initialize state
            state = TextState(document_id=document_id, images=images)
            
            # Create and run graph
            graph = self.create_graph()
            compiled_graph = graph.compile()
            
            # Execute graph
            result_state = compiled_graph.invoke(state)
            
            # Prepare response
            response = {
                "success": len(result_state.errors) == 0,
                "document_id": result_state.document_id,
                "text_summary": result_state.text_summary,
                "extracted_entities": result_state.extracted_entities,
                "ocr_results": [
                    {
                        "page_num": result.page_num,
                        "text_preview": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                        "average_confidence": result.average_confidence,
                        "engine_used": result.engine_used
                    }
                    for result in result_state.ocr_results
                ],
                "errors": result_state.errors
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return {
                "success": False,
                "document_id": document_id,
                "error": str(e)
            }

class GrokClient:
    """Client for Grok API"""
    
    def __init__(self):
        self.api_key = settings.GROK_API_KEY
        self.api_url = settings.GROK_API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, prompt: str) -> Any:
        """Generate response using Grok API"""
        try:
            payload = {
                "model": "grok-beta",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
        except Exception as e:
            logger.error(f"Grok API call failed: {e}")
            raise

class OllamaClient:
    """Client for Ollama"""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
    
    def generate(self, prompt: str) -> Any:
        """Generate response using Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
            
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise