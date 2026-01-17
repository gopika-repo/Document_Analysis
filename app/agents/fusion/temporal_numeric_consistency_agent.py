from typing import Dict, List, Any, Tuple
import re
from datetime import datetime
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class TemporalNumericConsistencyAgent:
    """Agent for checking temporal and numeric consistency across document"""
    
    def __init__(self):
        self.date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y',
            '%Y/%m/%d', '%d %b %Y', '%b %d, %Y'
        ]
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Check temporal and numeric consistency"""
        try:
            logger.info(f"Checking temporal/numeric consistency for {state.document_id}")
            
            consistency_checks = {
                "temporal_consistency": self._check_temporal_consistency(state),
                "numeric_consistency": self._check_numeric_consistency(state),
                "calculation_checks": self._check_calculations(state),
                "sequence_checks": self._check_sequences(state),
                "overall_consistency_score": 0.0
            }
            
            # Calculate overall consistency score
            consistency_checks["overall_consistency_score"] = self._calculate_consistency_score(
                consistency_checks
            )
            
            state.temporal_consistency = consistency_checks
            logger.info(f"Consistency check completed: score {consistency_checks['overall_consistency_score']:.2f}")
            
            return state
            
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            state.temporal_consistency = {"error": str(e), "overall_consistency_score": 0.3}
            return state
    
    def _check_temporal_consistency(self, state: ProcessingState) -> Dict[str, Any]:
        """Check temporal consistency (dates, time sequences)"""
        checks = {
            "date_count": 0,
            "parsed_dates": [],
            "date_issues": [],
            "temporal_order_issues": [],
            "year_consistency": True,
            "month_consistency": True
        }
        
        try:
            # Extract dates from entities
            dates = self._extract_dates_from_state(state)
            checks["date_count"] = len(dates)
            
            if dates:
                # Parse dates
                parsed_dates = []
                for date_str in dates:
                    parsed = self._parse_date(date_str)
                    if parsed:
                        parsed_dates.append(parsed)
                
                checks["parsed_dates"] = [d.strftime('%Y-%m-%d') for d in parsed_dates if d]
                
                # Check temporal order if multiple dates
                if len(parsed_dates) > 1:
                    sorted_dates = sorted(parsed_dates)
                    
                    # Check if dates are in chronological order (as they appear in doc)
                    if parsed_dates != sorted_dates:
                        checks["temporal_order_issues"].append(
                            f"Dates not in chronological order: {checks['parsed_dates']}"
                        )
                    
                    # Check year consistency
                    years = [d.year for d in parsed_dates]
                    if max(years) - min(years) > 10:  # More than 10 years difference
                        checks["year_consistency"] = False
                        checks["date_issues"].append(
                            f"Large year range detected: {min(years)} to {max(years)}"
                        )
        
        except Exception as e:
            logger.warning(f"Temporal consistency check failed: {e}")
            checks["date_issues"].append(f"Check error: {str(e)}")
        
        return checks
    
    def _extract_dates_from_state(self, state: ProcessingState) -> List[str]:
        """Extract dates from various sources in state"""
        dates = []
        
        # From extracted entities
        if hasattr(state, 'extracted_entities') and state.extracted_entities:
            if 'dates' in state.extracted_entities:
                dates.extend(state.extracted_entities['dates'])
        
        # From OCR text (fallback)
        if not dates and hasattr(state, 'ocr_results') and state.ocr_results:
            all_text = ""
            for ocr_result in state.ocr_results.values():
                if isinstance(ocr_result, dict) and 'text' in ocr_result:
                    all_text += ocr_result['text'] + " "
            
            # Simple date pattern matching
            date_patterns = [
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
                r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, all_text, re.IGNORECASE)
                dates.extend(matches)
        
        # Deduplicate
        return list(set(dates))
    
    def _parse_date(self, date_str: str):
        """Parse date string into datetime object"""
        for fmt in self.date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        # Try common variations
        try:
            # Remove common suffixes
            clean_str = re.sub(r'(st|nd|rd|th)', '', date_str, flags=re.IGNORECASE)
            for fmt in self.date_formats:
                try:
                    return datetime.strptime(clean_str, fmt)
                except:
                    continue
        except:
            pass
        
        return None
    
    def _check_numeric_consistency(self, state: ProcessingState) -> Dict[str, Any]:
        """Check numeric consistency (amounts, percentages)"""
        checks = {
            "amount_count": 0,
            "amounts": [],
            "percentage_count": 0,
            "percentages": [],
            "numeric_issues": [],
            "range_checks": {}
        }
        
        try:
            # Extract numeric values
            amounts = self._extract_amounts_from_state(state)
            percentages = self._extract_percentages_from_state(state)
            
            checks["amount_count"] = len(amounts)
            checks["amounts"] = amounts[:10]  # First 10 as sample
            checks["percentage_count"] = len(percentages)
            checks["percentages"] = percentages[:10]
            
            # Check for unrealistic values
            if amounts:
                numeric_amounts = self._parse_numeric_values(amounts)
                if numeric_amounts:
                    max_amount = max(numeric_amounts)
                    min_amount = min(numeric_amounts)
                    
                    checks["range_checks"]["amount_range"] = f"{min_amount} to {max_amount}"
                    
                    # Check for unusually large amounts
                    if max_amount > 1000000:  # 1 million
                        checks["numeric_issues"].append(
                            f"Unusually large amount detected: {max_amount}"
                        )
                    
                    # Check for negative amounts (might be errors)
                    if min_amount < 0:
                        checks["numeric_issues"].append(
                            f"Negative amount detected: {min_amount}"
                        )
            
            if percentages:
                numeric_percentages = self._parse_numeric_values(percentages)
                if numeric_percentages:
                    # Check for percentages > 100%
                    for pct in numeric_percentages:
                        if pct > 100:
                            checks["numeric_issues"].append(
                                f"Percentage exceeds 100%: {pct}%"
                            )
                        if pct < 0:
                            checks["numeric_issues"].append(
                                f"Negative percentage: {pct}%"
                            )
        
        except Exception as e:
            logger.warning(f"Numeric consistency check failed: {e}")
            checks["numeric_issues"].append(f"Check error: {str(e)}")
        
        return checks
    
    def _extract_amounts_from_state(self, state: ProcessingState) -> List[str]:
        """Extract monetary amounts from state"""
        amounts = []
        
        if hasattr(state, 'extracted_entities') and state.extracted_entities:
            if 'amounts' in state.extracted_entities:
                amounts.extend(state.extracted_entities['amounts'])
        
        return amounts
    
    def _extract_percentages_from_state(self, state: ProcessingState) -> List[str]:
        """Extract percentages from state"""
        percentages = []
        
        if hasattr(state, 'extracted_entities') and state.extracted_entities:
            if 'percentages' in state.extracted_entities:
                percentages.extend(state.extracted_entities['percentages'])
        
        return percentages
    
    def _parse_numeric_values(self, values: List[str]) -> List[float]:
        """Parse numeric values from strings"""
        numeric_values = []
        
        for value in values:
            try:
                # Extract numbers from string
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', value.replace(',', ''))
                if numbers:
                    numeric_values.append(float(numbers[0]))
            except:
                continue
        
        return numeric_values
    
    def _check_calculations(self, state: ProcessingState) -> Dict[str, Any]:
        """Check calculation consistency"""
        checks = {
            "total_checks": 0,
            "calculation_issues": [],
            "summation_checks": []
        }
        
        try:
            # Look for totals and sums in text
            if hasattr(state, 'ocr_results') and state.ocr_results:
                all_text = ""
                for ocr_result in state.ocr_results.values():
                    if isinstance(ocr_result, dict) and 'text' in ocr_result:
                        all_text += ocr_result['text'] + "\n"
                
                # Look for summation patterns
                sum_patterns = [
                    r'total.*?\$?(\d+(?:,\d+)*(?:\.\d{2})?)',
                    r'sum.*?\$?(\d+(?:,\d+)*(?:\.\d{2})?)',
                    r'amount.*?\$?(\d+(?:,\d+)*(?:\.\d{2})?)',
                ]
                
                for pattern in sum_patterns:
                    matches = re.findall(pattern, all_text, re.IGNORECASE)
                    if matches:
                        checks["summation_checks"].extend(matches)
                        checks["total_checks"] += len(matches)
        
        except Exception as e:
            logger.warning(f"Calculation check failed: {e}")
            checks["calculation_issues"].append(f"Check error: {str(e)}")
        
        return checks
    
    def _check_sequences(self, state: ProcessingState) -> Dict[str, Any]:
        """Check sequential patterns"""
        checks = {
            "sequence_issues": [],
            "pattern_checks": []
        }
        
        try:
            # Check for numbered sequences
            if hasattr(state, 'ocr_results') and state.ocr_results:
                all_text = ""
                for ocr_result in state.ocr_results.values():
                    if isinstance(ocr_result, dict) and 'text' in ocr_result:
                        all_text += ocr_result['text'] + "\n"
                
                # Look for numbered lists
                numbered_items = re.findall(r'(\d+)\.\s+[A-Z]', all_text)
                if numbered_items:
                    numbers = [int(n) for n in numbered_items]
                    
                    # Check if sequence is complete
                    if numbers:
                        expected_sequence = list(range(min(numbers), max(numbers) + 1))
                        if numbers != expected_sequence:
                            checks["sequence_issues"].append(
                                f"Incomplete numbered sequence: {numbers}"
                            )
        
        except Exception as e:
            logger.warning(f"Sequence check failed: {e}")
            checks["sequence_issues"].append(f"Check error: {str(e)}")
        
        return checks
    
    def _calculate_consistency_score(self, checks: Dict[str, Any]) -> float:
        """Calculate overall consistency score"""
        score_factors = []
        
        # Temporal consistency factor
        temporal_checks = checks.get("temporal_consistency", {})
        if temporal_checks.get("date_count", 0) > 0:
            if not temporal_checks.get("date_issues") and not temporal_checks.get("temporal_order_issues"):
                score_factors.append(0.9)
            else:
                issues_count = len(temporal_checks.get("date_issues", [])) + \
                             len(temporal_checks.get("temporal_order_issues", []))
                penalty = min(issues_count * 0.2, 0.6)
                score_factors.append(0.9 - penalty)
        
        # Numeric consistency factor
        numeric_checks = checks.get("numeric_consistency", {})
        if numeric_checks.get("amount_count", 0) > 0 or numeric_checks.get("percentage_count", 0) > 0:
            if not numeric_checks.get("numeric_issues"):
                score_factors.append(0.8)
            else:
                issues_count = len(numeric_checks.get("numeric_issues", []))
                penalty = min(issues_count * 0.15, 0.5)
                score_factors.append(0.8 - penalty)
        
        # Calculation consistency factor
        calculation_checks = checks.get("calculation_checks", {})
        if calculation_checks.get("total_checks", 0) > 0:
            if not calculation_checks.get("calculation_issues"):
                score_factors.append(0.7)
        
        # Sequence consistency factor
        sequence_checks = checks.get("sequence_checks", {})
        if not sequence_checks.get("sequence_issues"):
            score_factors.append(0.6)
        
        # Calculate average score
        if score_factors:
            return sum(score_factors) / len(score_factors)
        
        return 0.7  # Default consistency score