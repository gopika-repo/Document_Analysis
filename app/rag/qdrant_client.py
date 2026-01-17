from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
import uuid
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class QdrantManager:
    """Manager for Qdrant vector database operations"""
    
    def __init__(self):
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION
        self.vector_size = settings.VECTOR_SIZE
        self._connect()
    
    def _connect(self):
        """Connect to Qdrant"""
        try:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=30
            )
            logger.info(f"Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    async def initialize_collection(self) -> bool:
        """Initialize collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            return False
    
    def store_document(self, 
                      document_id: str,
                      text_embedding: List[float],
                      visual_embedding: Optional[List[float]] = None,
                      metadata: Dict[str, Any] = None) -> bool:
        """
        Store document embeddings in Qdrant
        
        Args:
            document_id: Unique document identifier
            text_embedding: Text embedding vector
            visual_embedding: Optional visual embedding vector
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            # Prepare point ID
            point_id = str(uuid.uuid4())
            
            # Prepare payload
            payload = {
                "document_id": document_id,
                "text_embedding_size": len(text_embedding),
                "has_visual_embedding": visual_embedding is not None
            }
            
            if metadata:
                payload.update(metadata)
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=text_embedding,
                payload=payload
            )
            
            # Store point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Stored document {document_id} in Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            return False
    
    def search_similar(self, 
                      query_embedding: List[float],
                      limit: int = 5,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            filters: Optional filters
            
        Returns:
            List of similar documents
        """
        try:
            # Prepare filter
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                qdrant_filter = Filter(must=conditions)
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=limit
            )
            
            # Format results
            results = []
            for hit in search_result:
                results.append({
                    "document_id": hit.payload.get("document_id"),
                    "score": hit.score,
                    "metadata": hit.payload
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete document from Qdrant
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Success status
        """
        try:
            # Find points for this document
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )]
                )
            )
            
            # Extract point IDs
            point_ids = [point.id for point in search_result[0]]
            
            if point_ids:
                # Delete points
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                logger.info(f"Deleted document {document_id} from Qdrant")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            
            return {
                "name": info.config.params.vectors.size,
                "vector_size": info.config.params.vectors.size,
                "points_count": info.points_count,
                "segments_count": info.segments_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}