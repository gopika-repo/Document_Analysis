from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import uuid
import shutil
from datetime import datetime

from app.config import settings
from app.agents.orchestrator import AgentOrchestrator
from app.rag.retriever import MultiModalRetriever
from app.utils.logger import setup_logger
from app.utils.error_handler import handle_api_error

logger = setup_logger(__name__)

router = APIRouter()

# Initialize components
orchestrator = AgentOrchestrator()
retriever = MultiModalRetriever()

# Store processing status (in production, use Redis or database)
processing_status = {}

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a document for processing
    
    Supports PDF and image files
    """
    try:
        logger.info(f"Upload request received for file: {file.filename}")
        
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Create upload directory
        upload_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(upload_dir, f"original{file_ext}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to {file_path}")
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            document_id,
            file_path
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document uploaded and processing started",
            "status_endpoint": f"/api/v1/status/{document_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process")
async def process_document(
    document_id: str,
    reprocess: bool = False
):
    """
    Trigger document processing
    
    Can be used to reprocess an already uploaded document
    """
    try:
        logger.info(f"Process request for document: {document_id}")
        
        # Check if document exists
        doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        if not os.path.exists(doc_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Find original file
        original_files = [
            f for f in os.listdir(doc_dir) 
            if f.startswith("original")
        ]
        
        if not original_files:
            raise HTTPException(
                status_code=404,
                detail=f"Original file not found for document {document_id}"
            )
        
        file_path = os.path.join(doc_dir, original_files[0])
        
        # Start processing
        result = await orchestrator.process_document(file_path)
        
        # Save result
        result_file = os.path.join(doc_dir, "processing_result.json")
        import json
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        # Update status
        processing_status[document_id] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False)
        }
        
        return {
            "success": True,
            "document_id": document_id,
            "processing_complete": True,
            "result_available": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        
        # Update status with error
        processing_status[document_id] = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
        
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_document(
    query_request: dict
):
    """
    Query processed documents
    
    Example request:
    {
        "document_id": "doc_123",
        "question": "What does the chart imply about revenue?"
    }
    """
    try:
        document_id = query_request.get("document_id")
        question = query_request.get("question")
        
        if not document_id or not question:
            raise HTTPException(
                status_code=400,
                detail="document_id and question are required"
            )
        
        logger.info(f"Query request: document={document_id}, question={question[:50]}...")
        
        # Load processing results
        doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        result_file = os.path.join(doc_dir, "processing_result.json")
        
        if not os.path.exists(result_file):
            raise HTTPException(
                status_code=404,
                detail=f"Processing results not found for document {document_id}"
            )
        
        import json
        with open(result_file, "r") as f:
            processing_results = json.load(f)
        
        # Extract relevant information for answering
        extracted_data = processing_results.get("extracted_data", {})
        
        # Simple query answering (in production, use LLM for better answering)
        answer = generate_answer(question, extracted_data)
        
        return {
            "success": True,
            "document_id": document_id,
            "question": question,
            "answer": answer,
            "confidence": 0.8,  # Placeholder
            "sources": ["extracted_data", "field_analysis"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{document_id}")
async def get_results(document_id: str):
    """
    Get processing results for a document
    """
    try:
        logger.info(f"Results request for document: {document_id}")
        
        doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        result_file = os.path.join(doc_dir, "processing_result.json")
        
        if not os.path.exists(result_file):
            raise HTTPException(
                status_code=404,
                detail=f"Results not found for document {document_id}"
            )
        
        import json
        with open(result_file, "r") as f:
            results = json.load(f)
        
        # Add status information
        status = processing_status.get(document_id, {"status": "unknown"})
        
        return {
            "success": True,
            "document_id": document_id,
            "status": status["status"],
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Results retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{document_id}")
async def get_status(document_id: str):
    """
    Get processing status for a document
    """
    try:
        status = processing_status.get(document_id, {"status": "not_found"})
        
        return {
            "document_id": document_id,
            "status": status["status"],
            "timestamp": status.get("timestamp"),
            "error": status.get("error")
        }
        
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/index")
async def index_document(
    document_id: str,
    background_tasks: BackgroundTasks
):
    """
    Index a processed document in the RAG system
    """
    try:
        logger.info(f"RAG index request for document: {document_id}")
        
        # Load processing results
        doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        result_file = os.path.join(doc_dir, "processing_result.json")
        
        if not os.path.exists(result_file):
            raise HTTPException(
                status_code=404,
                detail=f"Processing results not found for document {document_id}"
            )
        
        import json
        with open(result_file, "r") as f:
            processing_results = json.load(f)
        
        # Extract text content for indexing
        text_content = extract_text_for_indexing(processing_results)
        
        # Extract images if available
        images = []
        # In production, load actual images
        
        # Start background indexing
        background_tasks.add_task(
            index_document_background,
            document_id,
            text_content,
            images,
            processing_results
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document indexing started",
            "text_length": len(text_content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG indexing request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/search")
async def search_documents(
    search_request: dict
):
    """
    Search indexed documents
    
    Example request:
    {
        "query": "revenue chart contradiction",
        "query_type": "text",
        "limit": 5
    }
    """
    try:
        query = search_request.get("query")
        query_type = search_request.get("query_type", "text")
        limit = search_request.get("limit", 5)
        
        if not query:
            raise HTTPException(
                status_code=400,
                detail="query is required"
            )
        
        logger.info(f"RAG search: {query[:50]}...")
        
        # Perform search
        results = await retriever.search_documents(
            query=query,
            query_type=query_type,
            limit=limit
        )
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_answer(question: str, extracted_data: dict) -> str:
    """Generate answer from extracted data"""
    # Simple rule-based answering
    # In production, use LLM for better answering
    
    if "chart" in question.lower() and "revenue" in question.lower():
        fields = extracted_data.get("fields", {})
        
        # Look for chart-related fields
        chart_fields = [
            f for f in fields.keys() 
            if "chart" in f.lower()
        ]
        
        if chart_fields:
            return "Chart analysis indicates revenue trends based on visual data. Please check specific chart fields for details."
        else:
            return "No chart data found in the document."
    
    elif "table" in question.lower():
        return "Table data has been extracted and structured. Refer to table fields for specific values."
    
    else:
        return "Question answered based on extracted document data. For detailed analysis, please review specific fields."

def extract_text_for_indexing(processing_results: dict) -> str:
    """Extract text content for RAG indexing"""
    text_parts = []
    
    # Extract from text results
    text_summary = processing_results.get("agent_outputs", {}).get("text", {}).get("summary", {})
    if text_summary:
        text_parts.append(str(text_summary))
    
    # Extract from OCR results
    ocr_results = processing_results.get("agent_outputs", {}).get("text", {}).get("ocr_results", [])
    for ocr_result in ocr_results:
        if "text_preview" in ocr_result:
            text_parts.append(ocr_result["text_preview"])
    
    # Extract from fields
    fields = processing_results.get("extracted_data", {}).get("fields", {})
    for field_name, field_data in fields.items():
        text_parts.append(f"{field_name}: {field_data.get('value', '')}")
    
    return "\n".join(text_parts)

async def process_document_background(document_id: str, file_path: str):
    """Background task for document processing"""
    try:
        logger.info(f"Starting background processing for {document_id}")
        
        # Update status
        processing_status[document_id] = {
            "status": "processing",
            "timestamp": datetime.now().isoformat()
        }
        
        # Process document
        result = await orchestrator.process_document(file_path)
        
        # Save result
        doc_dir = os.path.dirname(file_path)
        result_file = os.path.join(doc_dir, "processing_result.json")
        
        import json
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        # Update status
        processing_status[document_id] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False)
        }
        
        logger.info(f"Background processing completed for {document_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {document_id}: {e}")
        
        processing_status[document_id] = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

async def index_document_background(document_id: str, text_content: str, 
                                   images: list, metadata: dict):
    """Background task for document indexing"""
    try:
        logger.info(f"Starting background indexing for {document_id}")
        
        success = await retriever.index_document(
            document_id=document_id,
            text_content=text_content,
            images=images,
            metadata=metadata
        )
        
        if success:
            logger.info(f"Background indexing completed for {document_id}")
        else:
            logger.error(f"Background indexing failed for {document_id}")
        
    except Exception as e:
        logger.error(f"Background indexing failed for {document_id}: {e}")