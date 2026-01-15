"""FAISS Vector Store Service for storing and searching embeddings."""

import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import pickle
import os
import logging
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing FAISS vector store."""
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize vector store service.
        
        Args:
            index_path: Path to save/load FAISS index. If None, uses in-memory only.
        """
        self.index_path = index_path or "vector_store.index"
        self.metadata_path = self.index_path.replace(".index", "_metadata.pkl")
        self._index: Optional[faiss.Index] = None
        self._metadata: List[Dict[str, Any]] = []
        self._dimension = settings.embedding_dim
        self._initialized = False
    
    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index."""
        # Use IndexFlatIP (Inner Product) for cosine similarity with normalized vectors
        # or IndexFlatL2 for L2 distance
        index = faiss.IndexFlatIP(self._dimension)
        logger.info(f"Created new FAISS index with dimension {self._dimension}")
        return index
    
    def initialize(self) -> None:
        """Initialize or load existing vector store."""
        if self._initialized:
            return
        
        try:
            # Try to load existing index
            if os.path.exists(self.index_path):
                logger.info(f"Loading existing vector store from {self.index_path}")
                self._index = faiss.read_index(self.index_path)
                
                # Load metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'rb') as f:
                        self._metadata = pickle.load(f)
                    logger.info(f"Loaded {len(self._metadata)} metadata entries")
                else:
                    self._metadata = []
                    logger.warning("Index found but metadata not found. Starting with empty metadata.")
            else:
                # Create new index
                logger.info("Creating new vector store")
                self._index = self._create_index()
                self._metadata = []
            
            self._initialized = True
            logger.info(f"Vector store initialized. Current size: {self.ntotal}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            # Fallback to new index
            self._index = self._create_index()
            self._metadata = []
            self._initialized = True
            raise
    
    def save(self) -> None:
        """Save index and metadata to disk."""
        if not self._initialized or self._index is None:
            logger.warning("Cannot save: vector store not initialized")
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else ".", exist_ok=True)
            
            # Save index
            faiss.write_index(self._index, self.index_path)
            logger.info(f"Saved FAISS index to {self.index_path}")
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self._metadata, f)
            logger.info(f"Saved metadata to {self.metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def add_vectors(
        self, 
        embeddings: np.ndarray, 
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Add vectors to the index.
        
        Args:
            embeddings: Embedding vectors of shape (n, dimension)
            texts: Original texts corresponding to embeddings
            metadata: Optional additional metadata for each text
            
        Returns:
            List of IDs assigned to the added vectors
        """
        if not self._initialized:
            self.initialize()
        
        if self._index is None:
            raise RuntimeError("Vector store not initialized")
        
        if len(embeddings) != len(texts):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings but {len(texts)} texts")
        
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype(np.float32)
        
        # Get current size before adding
        start_id = self.ntotal
        
        # Add to FAISS index
        self._index.add(embeddings)
        
        # Add metadata
        for i, text in enumerate(texts):
            meta = {
                "id": start_id + i,
                "text": text,
                "added_at": None  # Could add timestamp if needed
            }
            if metadata and i < len(metadata):
                meta.update(metadata[i])
            self._metadata.append(meta)
        
        logger.info(f"Added {len(embeddings)} vectors to store. Total: {self.ntotal}")
        return list(range(start_id, start_id + len(embeddings)))
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Search for top k similar vectors.
        
        Args:
            query_embedding: Query embedding vector of shape (dimension,)
            k: Number of top results to return
            
        Returns:
            Tuple of (distances, indices, metadata_list)
            - distances: Similarity scores (higher is better for IP)
            - indices: IDs of similar vectors
            - metadata_list: Metadata for each result
        """
        if not self._initialized:
            self.initialize()
        
        if self._index is None or self.ntotal == 0:
            return np.array([]), np.array([]), []
        
        # Ensure query is float32 and normalized
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search
        k = min(k, self.ntotal)  # Don't search for more than available
        distances, indices = self._index.search(query_embedding, k)
        
        # Get metadata for results
        results_metadata = []
        for idx in indices[0]:
            if 0 <= idx < len(self._metadata):
                results_metadata.append(self._metadata[idx])
            else:
                results_metadata.append({"id": int(idx), "text": None, "error": "Metadata not found"})
        
        return distances[0], indices[0], results_metadata
    
    def delete_by_ids(self, ids: List[int]) -> int:
        """
        Delete vectors by IDs (Note: FAISS doesn't support deletion directly,
        this is a placeholder for future implementation with IndexIDMap).
        
        Args:
            ids: List of IDs to delete
            
        Returns:
            Number of vectors deleted
        """
        logger.warning("Direct deletion not supported in current FAISS implementation")
        return 0
    
    def clear(self) -> None:
        """Clear all vectors from the store."""
        self._index = self._create_index()
        self._metadata = []
        logger.info("Vector store cleared")
    
    @property
    def ntotal(self) -> int:
        """Get total number of vectors in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        return self._initialized



