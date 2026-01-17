from typing import Dict, List, Any
import re
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class SemanticReasoningAgent:
    """Agent for semantic analysis and understanding of document content"""
    
    def __init__(self):
        self.positive_words = {
            'increase', 'growth', 'profit', 'gain', 'positive', 'success', 
            'improve', 'better', 'excellent', 'good', 'strong', 'rise'
        }
        
        self.negative_words = {
            'decrease', 'decline', 'loss', 'negative', 'fail', 'problem',
            'worse', 'poor', 'weak', 'drop', 'fall', 'issue'
        }
        
        self.uncertainty_words = {
            'estimate', 'approximately', 'about', 'roughly', 'maybe',
            'possibly', 'could', 'might', 'potential', 'projected'
        }
        
        self.financial_keywords = {
            'revenue', 'profit', 'expense', 'cost', 'income', 'earnings',
            'balance', 'statement', 'quarter', 'annual', 'financial'
        }
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Perform semantic analysis of document content"""
        try:
            logger.info(f"Performing semantic analysis for {state.document_id}")
            
            # Extract text for analysis
            text = self._extract_analysis_text(state)
            
            # Perform semantic analysis
            semantic_analysis = {
                "document_type": self._infer_document_type(text),
                "sentiment": self._analyze_sentiment(text),
                "key_themes": self._extract_key_themes(text),
                "uncertainty_level": self._assess_uncertainty(text),
                "financial_focus": self._assess_financial_focus(text),
                "summary": self._generate_summary(text),
                "confidence": self._calculate_semantic_confidence(text)
            }
            
            state.semantic_analysis = semantic_analysis
            logger.info(f"Semantic analysis completed: {semantic_analysis['document_type']}")
            
            return state
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            state.semantic_analysis = {
                "document_type": "unknown",
                "sentiment": "neutral",
                "confidence": 0.0,
                "error": str(e)
            }
            return state
    
    def _extract_analysis_text(self, state: ProcessingState) -> str:
        """Extract text for semantic analysis"""
        text = ""
        
        # Use OCR text if available
        if hasattr(state, 'ocr_results') and state.ocr_results:
            for ocr_result in state.ocr_results.values():
                if isinstance(ocr_result, dict) and 'text' in ocr_result:
                    text += ocr_result['text'] + "\n"
                elif isinstance(ocr_result, str):
                    text += ocr_result + "\n"
        
        # Use entity information if available
        if hasattr(state, 'extracted_entities') and state.extracted_entities:
            # Add entities as context
            for entity_type, entities in state.extracted_entities.items():
                if entities:
                    text += f"Found {entity_type}: {', '.join(entities[:5])}\n"
        
        return text[:5000]  # Limit text length for analysis
    
    def _infer_document_type(self, text: str) -> str:
        """Infer document type from content"""
        text_lower = text.lower()
        
        # Check for document type indicators
        indicators = {
            "financial_report": [
                'financial statement', 'balance sheet', 'income statement',
                'quarterly report', 'annual report', 'revenue', 'profit', 'earnings'
            ],
            "invoice": [
                'invoice', 'bill', 'payment due', 'amount', 'total due',
                'item', 'quantity', 'price', 'tax'
            ],
            "contract": [
                'agreement', 'contract', 'terms', 'conditions', 'party',
                'obligation', 'termination', 'effective date'
            ],
            "research_paper": [
                'abstract', 'introduction', 'methodology', 'results',
                'discussion', 'references', 'conclusion'
            ]
        }
        
        scores = {}
        for doc_type, keywords in indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[doc_type] = score
        
        # Find document type with highest score
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                for doc_type, score in scores.items():
                    if score == max_score:
                        return doc_type
        
        return "general_document"
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the document"""
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return {
                "overall": "neutral",
                "score": 0.0,
                "positive_count": 0,
                "negative_count": 0
            }
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        if sentiment_score > 0.2:
            overall = "positive"
        elif sentiment_score < -0.2:
            overall = "negative"
        else:
            overall = "neutral"
        
        return {
            "overall": overall,
            "score": sentiment_score,
            "positive_count": positive_count,
            "negative_count": negative_count
        }
    
    def _extract_key_themes(self, text: str) -> List[str]:
        """Extract key themes from text"""
        themes = []
        text_lower = text.lower()
        
        # Common theme categories
        theme_categories = {
            "financial": ['revenue', 'profit', 'cost', 'investment', 'budget'],
            "performance": ['growth', 'decline', 'improvement', 'target', 'goal'],
            "temporal": ['quarter', 'annual', 'monthly', 'year', 'period'],
            "comparative": ['vs', 'compared', 'versus', 'relative', 'difference']
        }
        
        for theme_name, keywords in theme_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme_name)
        
        return themes
    
    def _assess_uncertainty(self, text: str) -> float:
        """Assess level of uncertainty in document"""
        words = text.lower().split()
        uncertainty_words = sum(1 for word in words if word in self.uncertainty_words)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        return min(uncertainty_words / (total_words / 100), 1.0)  # Normalize to 0-1
    
    def _assess_financial_focus(self, text: str) -> float:
        """Assess financial focus of document"""
        words = text.lower().split()
        financial_words = sum(1 for word in words if word in self.financial_keywords)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        return min(financial_words / (total_words / 100), 1.0)  # Normalize to 0-1
    
    def _generate_summary(self, text: str) -> str:
        """Generate a simple summary of the document"""
        # Extract first few sentences for summary
        sentences = re.split(r'[.!?]+', text)
        
        # Filter out very short sentences
        meaningful_sentences = [s.strip() for s in sentences if len(s.split()) > 5]
        
        if not meaningful_sentences:
            return "Document content could not be summarized."
        
        # Take first 3 meaningful sentences as summary
        summary = ". ".join(meaningful_sentences[:3]) + "."
        
        # Add document type inference
        doc_type = self._infer_document_type(text)
        if doc_type != "general_document":
            summary = f"This appears to be a {doc_type.replace('_', ' ')}. {summary}"
        
        return summary
    
    def _calculate_semantic_confidence(self, text: str) -> float:
        """Calculate confidence in semantic analysis"""
        confidence_factors = []
        
        # Text length factor
        word_count = len(text.split())
        if word_count > 500:
            confidence_factors.append(0.9)
        elif word_count > 100:
            confidence_factors.append(0.7)
        elif word_count > 50:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        # Entity richness factor (if we have entity info)
        if hasattr(self, '_entity_count'):
            entity_richness = min(self._entity_count / 10, 1.0)
            confidence_factors.append(entity_richness * 0.3)
        
        # Theme clarity factor
        themes = self._extract_key_themes(text)
        if themes:
            confidence_factors.append(0.2)
        
        # Calculate average confidence
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        return 0.5