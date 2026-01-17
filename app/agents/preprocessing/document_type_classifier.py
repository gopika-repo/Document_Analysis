import re
from typing import Dict, Any
from langgraph.graph import END
from app.core.models import ProcessingState, DocumentType
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentTypeClassifier:
    """Agent for classifying document type"""
    
    def __init__(self):
        self.keywords = {
            DocumentType.FINANCIAL_REPORT: [
                "financial", "revenue", "profit", "balance", "statement",
                "quarterly", "annual", "earnings", "income", "expense"
            ],
            DocumentType.INVOICE: [
                "invoice", "bill", "payment", "due", "amount",
                "tax", "total", "item", "quantity", "price"
            ],
            DocumentType.CONTRACT: [
                "agreement", "contract", "terms", "conditions",
                "party", "effective", "termination", "obligation"
            ],
            DocumentType.FORM: [
                "form", "application", "section", "checkbox",
                "signature", "date", "name", "address"
            ],
            DocumentType.RESEARCH_PAPER: [
                "abstract", "introduction", "methodology", "results",
                "discussion", "references", "figure", "table"
            ]
        }
    
    async def classify_document(self, state: ProcessingState) -> ProcessingState:
        """Classify document type based on content and structure"""
        try:
            logger.info(f"Classifying document type for {state.document_id}")
            
            # Get text from first few pages for classification
            sample_text = ""
            if hasattr(state, 'ocr_results') and state.ocr_results:
                for idx in range(min(3, len(state.ocr_results))):
                    if idx in state.ocr_results:
                        sample_text += state.ocr_results[idx].get('text', '')
            
            # Check visual elements
            visual_clues = self._analyze_visual_elements(state)
            
            # Combine text and visual analysis
            doc_type = self._determine_document_type(sample_text, visual_clues)
            
            state.document_type = doc_type
            logger.info(f"Document classified as: {doc_type}")
            
            return state
            
        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            state.document_type = DocumentType.UNKNOWN
            state.errors.append(f"Classification error: {str(e)}")
            return state
    
    def _analyze_visual_elements(self, state: ProcessingState) -> Dict[str, Any]:
        """Analyze visual elements for document type clues"""
        clues = {
            "has_tables": False,
            "has_charts": False,
            "has_signatures": False,
            "has_logos": False,
            "layout_complexity": 0
        }
        
        if hasattr(state, 'visual_elements') and state.visual_elements:
            for page_elements in state.visual_elements.values():
                for element in page_elements:
                    if element.element_type == "table":
                        clues["has_tables"] = True
                        clues["layout_complexity"] += 1
                    elif element.element_type == "chart":
                        clues["has_charts"] = True
                        clues["layout_complexity"] += 2
                    elif element.element_type == "signature":
                        clues["has_signatures"] = True
                    elif element.element_type == "logo":
                        clues["has_logos"] = True
        
        return clues
    
    def _determine_document_type(self, text: str, visual_clues: Dict[str, Any]) -> DocumentType:
        """Determine document type based on analysis"""
        scores = {doc_type: 0.0 for doc_type in DocumentType}
        
        # Text-based scoring
        text_lower = text.lower()
        for doc_type, keywords in self.keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            scores[doc_type] = score / max(len(keywords), 1)
        
        # Visual clue adjustments
        if visual_clues["has_charts"] and visual_clues["has_tables"]:
            scores[DocumentType.FINANCIAL_REPORT] += 0.3
            scores[DocumentType.RESEARCH_PAPER] += 0.2
        
        if visual_clues["has_signatures"]:
            scores[DocumentType.CONTRACT] += 0.3
            scores[DocumentType.FORM] += 0.2
        
        if visual_clues["layout_complexity"] > 5:
            scores[DocumentType.RESEARCH_PAPER] += 0.2
            scores[DocumentType.FINANCIAL_REPORT] += 0.1
        
        # Find highest scoring type
        max_score = 0
        best_type = DocumentType.UNKNOWN
        
        for doc_type, score in scores.items():
            if score > max_score:
                max_score = score
                best_type = doc_type
        
        # If score is too low, return MIXED
        if max_score < 0.3:
            return DocumentType.MIXED
        
        return best_type