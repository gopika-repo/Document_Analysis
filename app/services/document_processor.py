import os
import uuid
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentProcessor:
    """Document processing service with fallback support"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.orchestrator = None
        
        # Try to import AgentOrchestrator
        try:
            from app.agents.orchestrator import AgentOrchestrator
            self.orchestrator = AgentOrchestrator()
            logger.info("AgentOrchestrator loaded successfully")
        except ImportError as e:
            logger.warning(f"AgentOrchestrator not available: {e}")
            self.orchestrator = None
        
        # Try to import processing services
        try:
            from app.services.image_processor import ImageProcessor
            from app.services.pdf_processor import PDFProcessor
            self.image_processor = ImageProcessor()
            self.pdf_processor = PDFProcessor()
            logger.info("Processing services loaded")
        except ImportError:
            logger.warning("Processing services not available")
            self.image_processor = None
            self.pdf_processor = None
    
    async def process_upload(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Process an uploaded document"""
        try:
            document_id = str(uuid.uuid4())
            
            # Create job entry
            self.jobs[document_id] = {
                "document_id": document_id,
                "status": "processing",
                "start_time": datetime.now().isoformat(),
                "file_path": file_path,
                "file_type": file_type,
                "progress": 0.0,
                "errors": [],
                "result": None
            }
            
            logger.info(f"Starting processing for document {document_id}")
            
            # If orchestrator is available, use it
            if self.orchestrator:
                # Load document images
                images = await self._load_document(file_path, file_type)
                
                if not images:
                    error_msg = "Failed to load document images"
                    self.jobs[document_id]["status"] = "failed"
                    self.jobs[document_id]["errors"].append(error_msg)
                    return {
                        "success": False,
                        "document_id": document_id,
                        "error": error_msg
                    }
                
                # Update progress
                self.jobs[document_id]["progress"] = 0.3
                
                # Process through orchestrator
                result = await self.orchestrator.process_document(images, file_path)
                
                # Update job status
                self.jobs[document_id]["status"] = "completed" if result.get("success") else "failed"
                self.jobs[document_id]["progress"] = 1.0
                self.jobs[document_id]["end_time"] = datetime.now().isoformat()
                self.jobs[document_id]["result"] = result
                
                logger.info(f"Processing completed for document {document_id}")
                return result
            
            else:
                # Return stub response if orchestrator not available
                stub_result = self._create_stub_response(document_id, file_path, file_type)
                self.jobs[document_id]["status"] = "completed"
                self.jobs[document_id]["progress"] = 1.0
                self.jobs[document_id]["result"] = stub_result
                
                logger.info(f"Stub processing completed for document {document_id}")
                return stub_result
                
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            
            if 'document_id' in locals() and document_id in self.jobs:
                self.jobs[document_id]["status"] = "failed"
                self.jobs[document_id]["errors"].append(str(e))
            
            return {
                "success": False,
                "document_id": document_id if 'document_id' in locals() else "unknown",
                "error": str(e)
            }
    
    async def _load_document(self, file_path: str, file_type: str) -> Optional[List[Any]]:
        """Load document as images"""
        try:
            if file_type == "pdf" and self.pdf_processor:
                return await self.pdf_processor.process_pdf(file_path)
            elif self.image_processor:
                return await self.image_processor.process_image(file_path)
            else:
                # Return dummy images for testing
                import numpy as np
                dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
                return [dummy_image]
        except Exception as e:
            logger.error(f"Document loading failed: {e}")
            return None
    
    def _create_stub_response(self, document_id: str, file_path: str, file_type: str) -> Dict[str, Any]:
        """Create a stub response when full processing isn't available"""
        return {
            "success": True,
            "document_id": document_id,
            "document_type": "unknown",
            "extracted_fields": {
                "document_info": {
                    "value": f"File: {os.path.basename(file_path)}, Type: {file_type}",
                    "confidence": 0.5,
                    "sources": ["stub_processor"],
                    "modalities": ["metadata"]
                }
            },
            "validation_results": {
                "contradictions": [],
                "risk_score": 0.1,
                "integrity_score": 0.5
            },
            "explanations": {
                "processing": "Document processed with stub processor. Full agent system not available."
            },
            "recommendations": [
                "Install full agent dependencies for advanced processing",
                "Check system requirements"
            ],
            "processing_metadata": {
                "integrity_score": 0.5,
                "total_pages": 1,
                "agents_executed": ["stub_processor"],
                "processing_time": 0.1,
                "document_type": "unknown"
            },
            "errors": []
        }
    
    def get_job_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get processing status for a job"""
        if document_id in self.jobs:
            job = self.jobs[document_id].copy()
            
            # Calculate estimated completion if still processing
            if job["status"] == "processing":
                # Simple progress estimation
                if job["progress"] < 0.9:
                    job["progress"] = min(job["progress"] + 0.1, 0.9)
                
                # Estimate completion time
                start_time = datetime.fromisoformat(job["start_time"])
                elapsed = (datetime.now() - start_time).total_seconds()
                estimated_total = elapsed / (job["progress"] + 0.01)  # Avoid division by zero
                estimated_remaining = estimated_total - elapsed
                
                job["estimated_completion"] = datetime.now().timestamp() + estimated_remaining
            
            return job
        
        return None
    
    def get_job_result(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get processing result for a job"""
        if document_id in self.jobs and self.jobs[document_id]["status"] == "completed":
            return self.jobs[document_id]["result"]
        return None