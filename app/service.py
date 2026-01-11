"""BGE-M3 Embedding Service implementation."""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import logging

from .config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for BGE-M3 embeddings."""
    
    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._model_loaded: bool = False
    
    def load_model(self) -> None:
        """Load the BGE-M3 model."""
        if self._model is not None:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info(f"Loading embedding model: {settings.model_name}")
            self._model = SentenceTransformer(settings.model_name)
            self._model_loaded = True
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._model_loaded = False
            raise
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded and self._model is not None
    
    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            normalize: Whether to normalize embeddings (default: True)
            
        Returns:
            Embeddings as numpy array of shape (n_texts, embedding_dim)
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        if not texts:
            return np.array([])
        
        try:
            # Encode with optional normalization
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            
            # Ensure float32 for compatibility
            embeddings = embeddings.astype(np.float32)
            
            logger.debug(f"Encoded {len(texts)} texts to embeddings of shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single text to embedding.
        
        Args:
            text: Text string to encode
            normalize: Whether to normalize embedding (default: True)
            
        Returns:
            Embedding as numpy array of shape (embedding_dim,)
        """
        embeddings = self.encode([text], normalize=normalize)
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return settings.embedding_dim
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return settings.model_name


