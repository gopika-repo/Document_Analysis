from typing import Dict, List, Any
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class SignatureVerificationAgent:
    """Agent for verifying signatures in documents"""
    
    def __init__(self):
        self.signature_templates = {}  # Would be loaded from database in production
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        """Verify signatures in document"""
        try:
            logger.info(f"Verifying signatures for {state.document_id}")
            
            signature_results = {
                "signatures_found": 0,
                "verified_signatures": [],
                "unverified_signatures": [],
                "verification_confidence": 0.0
            }
            
            # Find signature elements
            signatures = self._find_signature_elements(state)
            
            for sig in signatures:
                verification = self._verify_signature(sig, state)
                
                if verification["is_verified"]:
                    signature_results["verified_signatures"].append(verification)
                else:
                    signature_results["unverified_signatures"].append(verification)
            
            signature_results["signatures_found"] = len(signatures)
            
            # Calculate overall confidence
            if signature_results["signatures_found"] > 0:
                verified_count = len(signature_results["verified_signatures"])
                signature_results["verification_confidence"] = verified_count / signature_results["signatures_found"]
            
            state.signature_verification = signature_results
            logger.info(f"Signature verification completed: {signature_results['signatures_found']} found")
            
            return state
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            state.signature_verification = {"error": str(e)}
            return state
    
    def _find_signature_elements(self, state: ProcessingState) -> List[Dict[str, Any]]:
        """Find signature elements in the document"""
        signatures = []
        
        # Check visual elements
        if hasattr(state, 'visual_elements') and state.visual_elements:
            for page_num, elements in state.visual_elements.items():
                for element in elements:
                    if hasattr(element, 'element_type') and element.element_type == 'signature':
                        signatures.append({
                            "page": page_num,
                            "bbox": element.bbox if hasattr(element, 'bbox') else [],
                            "confidence": element.confidence if hasattr(element, 'confidence') else 0.0,
                            "metadata": element.metadata if hasattr(element, 'metadata') else {}
                        })
        
        # Also check text for signature indicators
        if hasattr(state, 'ocr_results') and state.ocr_results:
            for page_num, ocr_result in state.ocr_results.items():
                if isinstance(ocr_result, dict) and 'text' in ocr_result:
                    text = ocr_result['text'].lower()
                    if 'signature' in text or 'signed' in text or 'authorized' in text:
                        signatures.append({
                            "page": page_num,
                            "type": "text_indicator",
                            "confidence": 0.6,
                            "text_context": text[:100]  # First 100 chars
                        })
        
        return signatures
    
    def _verify_signature(self, signature: Dict[str, Any], state: ProcessingState) -> Dict[str, Any]:
        """Verify a single signature"""
        verification = {
            "page": signature.get("page", 0),
            "is_verified": False,
            "verification_method": "unknown",
            "confidence": 0.0,
            "details": {}
        }
        
        try:
            # Check if signature has expected properties
            if "bbox" in signature and len(signature["bbox"]) == 4:
                # In production, this would compare with known signature templates
                # For now, use a simple heuristic
                bbox = signature["bbox"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Heuristic: Signatures are usually wider than tall
                aspect_ratio = width / height if height > 0 else 1
                
                if 1.5 < aspect_ratio < 4.0:  # Reasonable signature aspect ratio
                    verification["is_verified"] = True
                    verification["verification_method"] = "aspect_ratio_heuristic"
                    verification["confidence"] = min(signature.get("confidence", 0.5) * 0.8, 0.7)
                    verification["details"] = {
                        "aspect_ratio": aspect_ratio,
                        "width": width,
                        "height": height
                    }
                else:
                    verification["is_verified"] = False
                    verification["verification_method"] = "aspect_ratio_failed"
                    verification["confidence"] = 0.3
                    
            elif signature.get("type") == "text_indicator":
                # Text-based signature indicator
                verification["is_verified"] = True
                verification["verification_method"] = "text_indicator"
                verification["confidence"] = 0.6
                verification["details"] = {"text_context": signature.get("text_context", "")}
        
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            verification["details"]["error"] = str(e)
        
        return verification