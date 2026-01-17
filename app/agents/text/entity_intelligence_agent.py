from typing import Dict, List, Any
import re
from datetime import datetime
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EntityIntelligenceAgent:
    """Agent for extracting and normalizing entities from text"""
    
    def __init__(self):
        # Entity patterns
        self.date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # DD-MM-YYYY
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',    # YYYY-MM-DD
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',  # 15 Jan 2024
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}'  # January 15, 2024
        ]
        
        self.amount_patterns = [
            r'\$\d+(?:,\d+)*(?:\.\d{2})?',      # $1,234.56
            r'₹\d+(?:,\d+)*(?:\.\d{2})?',       # ₹1,234.56
            r'€\d+(?:,\d+)*(?:\.\d{2})?',       # €1,234.56
            r'\d+(?:,\d+)*(?:\.\d{2})?\s*(?:USD|EUR|GBP|INR)',  # 1234.56 USD
        ]
        
        self.percentage_patterns = [
            r'\d+(?:\.\d+)?\s*%',               # 15.5%
            r'\d+(?:\.\d+)?\s*percent',         # 15.5 percent
        ]
        
        self.phone_patterns = [
            r'\+\d{1,3}\s?\d{5,15}',            # +91 1234567890
            r'\(\d{3}\)\s?\d{3}[-]?\d{4}',      # (123) 456-7890
            r'\d{3}[-.]?\d{3}[-.]?\d{4}',       # 123-456-7890
        ]
        
        self.email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Extract entities from OCR text"""
        try:
            logger.info(f"Extracting entities for {state.document_id}")
            
            extracted_entities = {
                "dates": [],
                "amounts": [],
                "percentages": [],
                "phone_numbers": [],
                "emails": [],
                "companies": [],
                "names": [],
                "addresses": [],
            }
            
            # Extract text from OCR results
            all_text = self._extract_all_text(state)
            
            # Extract entities
            extracted_entities["dates"] = self._extract_dates(all_text)
            extracted_entities["amounts"] = self._extract_amounts(all_text)
            extracted_entities["percentages"] = self._extract_percentages(all_text)
            extracted_entities["phone_numbers"] = self._extract_phones(all_text)
            extracted_entities["emails"] = self._extract_emails(all_text)
            extracted_entities["companies"] = self._extract_companies(all_text)
            extracted_entities["names"] = self._extract_names(all_text)
            
            # Normalize and deduplicate
            for entity_type in extracted_entities:
                extracted_entities[entity_type] = self._normalize_entities(
                    extracted_entities[entity_type], entity_type
                )
            
            state.extracted_entities = extracted_entities
            logger.info(f"Extracted {sum(len(v) for v in extracted_entities.values())} entities")
            
            return state
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            state.extracted_entities = {}
            return state
    
    def _extract_all_text(self, state: ProcessingState) -> str:
        """Extract all text from OCR results"""
        all_text = ""
        
        if hasattr(state, 'ocr_results') and state.ocr_results:
            for ocr_result in state.ocr_results.values():
                if isinstance(ocr_result, dict) and 'text' in ocr_result:
                    all_text += ocr_result['text'] + "\n"
                elif isinstance(ocr_result, str):
                    all_text += ocr_result + "\n"
        
        return all_text
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text"""
        dates = []
        
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return dates
    
    def _extract_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts from text"""
        amounts = []
        
        for pattern in self.amount_patterns:
            matches = re.findall(pattern, text)
            amounts.extend(matches)
        
        return amounts
    
    def _extract_percentages(self, text: str) -> List[str]:
        """Extract percentages from text"""
        percentages = []
        
        for pattern in self.percentage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            percentages.extend(matches)
        
        return percentages
    
    def _extract_phones(self, text: str) -> List[str]:
        """Extract phone numbers from text"""
        phones = []
        
        for pattern in self.phone_patterns:
            matches = re.findall(pattern, text)
            phones.extend(matches)
        
        return phones
    
    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        return re.findall(self.email_pattern, text)
    
    def _extract_companies(self, text: str) -> List[str]:
        """Extract company names from text"""
        companies = []
        
        # Look for common company suffixes
        company_patterns = [
            r'([A-Z][a-zA-Z\s&]+)\s+(?:Inc|Ltd|LLC|Corp|Corporation|Company|Co\.)',
            r'(?:Company|Corporation)\s+([A-Z][a-zA-Z\s&]+)'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            companies.extend([m.strip() for m in matches])
        
        return companies
    
    def _extract_names(self, text: str) -> List[str]:
        """Extract person names from text"""
        names = []
        
        # Simple pattern for names (Title + First + Last)
        name_pattern = r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})'
        matches = re.findall(name_pattern, text)
        
        for match in matches:
            # Filter out common false positives
            if not any(word.lower() in ['company', 'corporation', 'ltd', 'inc'] for word in match.split()):
                names.append(match.strip())
        
        return names
    
    def _normalize_entities(self, entities: List[str], entity_type: str) -> List[str]:
        """Normalize and deduplicate entities"""
        normalized = []
        seen = set()
        
        for entity in entities:
            # Normalize based on entity type
            if entity_type == "dates":
                norm_entity = self._normalize_date(entity)
            elif entity_type == "amounts":
                norm_entity = self._normalize_amount(entity)
            else:
                norm_entity = entity.strip()
            
            if norm_entity and norm_entity not in seen:
                normalized.append(norm_entity)
                seen.add(norm_entity)
        
        return normalized
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string"""
        try:
            # Try to parse and standardize
            for fmt in ['%d-%m-%Y', '%m/%d/%Y', '%Y-%m-%d', '%d %b %Y']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d')
                except:
                    continue
        except:
            pass
        
        return date_str
    
    def _normalize_amount(self, amount_str: str) -> str:
        """Normalize amount string"""
        # Remove extra spaces and normalize currency symbols
        amount = amount_str.strip()
        
        # Standardize currency symbols
        currency_map = {
            '₹': 'INR ',
            '€': 'EUR ',
            '£': 'GBP ',
        }
        
        for symbol, replacement in currency_map.items():
            if amount.startswith(symbol):
                amount = replacement + amount[1:]
                break
        
        return amount