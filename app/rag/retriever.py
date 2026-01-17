from typing import List, Dict, Any, Optional
from app.rag.embeddings import EmbeddingEngine
from app.rag.qdrant_client import QdrantManager
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class MultiModalRetriever:
    """Multi-modal retriever for document search"""
    
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.qdrant_manager = QdrantManager()
    
    async def index_document(self, 
                           document_id: str,
                           text_content: str,
                           images: Optional[List[Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Index a document in the vector database
        
        Args:
            document_id: Unique document identifier
            text_content: Text content of the document
            images: Optional list of document images
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Indexing document {document_id}")
            
            # Generate embeddings
            text_embedding = self.embedding_engine.generate_text_embeddings(text_content)
            
            visual_embedding = None
            if images and len(images) > 0:
                visual_embedding = self.embedding_engine.generate_visual_embeddings(images[:1])[0]
            
            # Prepare metadata
            doc_metadata = {
                "document_id": document_id,
                "text_length": len(text_content),
                "has_images": images is not None and len(images) > 0
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            # Store in Qdrant
            success = self.qdrant_manager.store_document(
                document_id=document_id,
                text_embedding=text_embedding.tolist(),
                visual_embedding=visual_embedding.tolist() if visual_embedding is not None else None,
                metadata=doc_metadata
            )
            
            if success:
                logger.info(f"Document {document_id} indexed successfully")
            else:
                logger.error(f"Failed to index document {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return False
    
    async def search_documents(self, 
                             query: str,
                             query_type: str = "text",
                             filters: Optional[Dict[str, Any]] = None,
                             limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents
        
        Args:
            query: Search query
            query_type: Type of query ("text", "visual", "multi_modal")
            filters: Optional filters
            limit: Number of results
            
        Returns:
            List of matching documents
        """
        try:
            logger.info(f"Searching documents with query: {query[:50]}...")
            
            # Generate query embedding based on type
            if query_type == "text":
                query_embedding = self.embedding_engine.generate_text_embeddings(query)[0]
            elif query_type == "visual":
                # For visual queries, we'd need an image input
                # This is simplified - in production, you'd handle image queries differently
                query_embedding = self.embedding_engine.generate_text_embeddings(
                    f"Visual query: {query}"
                )[0]
            else:  # multi_modal
                query_embedding = self.embedding_engine.generate_text_embeddings(query)[0]
            
            # Search in Qdrant
            results = self.qdrant_manager.search_similar(
                query_embedding=query_embedding.tolist(),
                filters=filters,
                limit=limit
            )
            
            logger.info(f"Found {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []
    
    async def cross_modal_retrieval(self,
                                  text_query: Optional[str] = None,
                                  image_query: Optional[Any] = None,
                                  limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform cross-modal retrieval
        
        Args:
            text_query: Text query
            image_query: Image query
            limit: Number of results
            
        Returns:
            List of matching documents
        """
        try:
            logger.info("Performing cross-modal retrieval")
            
            # Generate embeddings for both modalities
            text_embedding = None
            if text_query:
                text_embedding = self.embedding_engine.generate_text_embeddings(text_query)[0]
            
            visual_embedding = None
            if image_query is not None:
                visual_embedding = self.embedding_engine.generate_visual_embeddings([image_query])[0]
            
            # For cross-modal, we can search with either embedding
            # In a more advanced system, you'd fuse them
            query_embedding = text_embedding if text_embedding is not None else visual_embedding
            
            if query_embedding is None:
                logger.error("No query provided for cross-modal retrieval")
                return []
            
            # Search
            results = self.qdrant_manager.search_similar(
                query_embedding=query_embedding.tolist(),
                limit=limit
            )
            
            # Filter or re-rank based on multi-modal relevance
            filtered_results = self._filter_cross_modal_results(
                results, text_query, image_query
            )
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Cross-modal retrieval failed: {e}")
            return []
    
    def _filter_cross_modal_results(self,
                                   results: List[Dict[str, Any]],
                                   text_query: Optional[str],
                                   image_query: Optional[Any]) -> List[Dict[str, Any]]:
        """Filter results for cross-modal relevance"""
        # Simplified filtering - in production, implement proper cross-modal scoring
        
        filtered = []
        for result in results:
            # Check if document has both text and visual content
            has_text = result["metadata"].get("text_length", 0) > 0
            has_images = result["metadata"].get("has_images", False)
            
            # Prefer documents that match both modalities
            if text_query and image_query:
                if has_text and has_images:
                    result["cross_modal_score"] = result["score"] * 1.2
                    filtered.append(result)
            elif text_query:
                if has_text:
                    filtered.append(result)
            elif image_query:
                if has_images:
                    filtered.append(result)
        
        return filtered
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from index"""
        try:
            success = self.qdrant_manager.delete_document(document_id)
            
            if success:
                logger.info(f"Document {document_id} deleted from index")
            else:
                logger.error(f"Failed to delete document {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.qdrant_manager.get_collection_stats()
            return {
                "vector_database": "Qdrant",
                "collection_name": self.qdrant_manager.collection_name,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}