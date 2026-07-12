"""
Vector Store module — Qdrant (cloud) manager.
Handles semantic search for job matching via embeddings using FastEmbed.
"""

from typing import Optional
from qdrant_client import QdrantClient
import config


class VectorStoreManager:
    """Manages Qdrant vector store for semantic job search."""

    def __init__(self):
        self.url = config.QDRANT_URL or config._get_config("QDRANT_URL", "")
        self.api_key = config.QDRANT_API_KEY or config._get_config("QDRANT_API_KEY", "")
        self.collection_name = config.COLLECTION_NAME
        
        # Connect to Qdrant Cloud
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )

    def add_documents(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        """
        Add job documents to the vector store.
        Automatically generates embeddings using Qdrant's fastembed client-side logic.
        """
        # Upload documents in batches
        import uuid
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            
            # Qdrant accepts UUIDs or unsigned integers.
            # Convert arbitrary string IDs (like "job_0") to deterministic UUIDs.
            processed_ids = [
                str(uuid.uuid5(uuid.NAMESPACE_DNS, str(item_id)))
                for item_id in batch_ids
            ]
                
            self.client.add(
                collection_name=self.collection_name,
                documents=batch_docs,
                metadata=batch_metas,
                ids=processed_ids,
            )

    def search_similar_jobs(self, query_text: str, top_k: int = 10) -> list[dict]:
        """
        Semantic search: find jobs most similar to the query text.
        Returns list of {id, document, metadata, similarity_score}.
        """
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                query_text=query_text,
                limit=top_k,
            )
            
            jobs = []
            for res in results:
                # Convert similarity score (typically between 0 and 1) to percentage
                score = res.score
                if score < 0:
                    score = 0
                elif score > 1:
                    score = 1
                    
                jobs.append({
                    "id": str(res.id),
                    "document": res.document,
                    "metadata": res.metadata,
                    "similarity_score": round(score * 100, 1),
                })
            return jobs
        except Exception as e:
            print(f"Error querying Qdrant: {e}")
            return []

    def match_cv_to_jobs(self, cv_text: str, top_k: int = 10) -> list[dict]:
        """
        Match CV content against job listings.
        Returns ranked list of matching jobs with similarity scores.
        """
        return self.search_similar_jobs(cv_text, top_k=top_k)

    def get_collection_count(self) -> int:
        """Get total number of documents in the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception:
            return 0

    def reset_collection(self):
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
