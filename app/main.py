"""BGE-M3 Embedding Service - FastAPI application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import logging

from .service import EmbeddingService
from .vector_store import VectorStoreService
from .config import settings

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BGE-M3 Embedding Service",
    version="1.0.0",
    description="Microservice for BGE-M3 text embeddings"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể config sau
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding service
embedding_service = EmbeddingService()

# Initialize vector store service
vector_store = VectorStoreService(index_path=settings.vector_store_path)


# Request/Response models
class EncodeRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of texts to encode")
    normalize: bool = Field(True, description="Whether to normalize embeddings")


class EncodeSingleRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to encode")
    normalize: bool = Field(True, description="Whether to normalize embeddings")


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimension: int = Field(..., description="Embedding dimension")
    count: int = Field(..., description="Number of embeddings")


class SingleEmbeddingResponse(BaseModel):
    embedding: List[float] = Field(..., description="Embedding vector")
    dimension: int = Field(..., description="Embedding dimension")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    dimension: int


class InfoResponse(BaseModel):
    model_name: str
    dimension: int
    description: str


class StoreRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of texts to embed and store")
    normalize: bool = Field(True, description="Whether to normalize embeddings")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="Optional metadata for each text")


class StoreResponse(BaseModel):
    ids: List[int] = Field(..., description="IDs of stored vectors")
    count: int = Field(..., description="Number of vectors stored")
    total_vectors: int = Field(..., description="Total vectors in store")


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Query text to search for")
    k: int = Field(10, ge=1, le=100, description="Number of top results to return")
    normalize: bool = Field(True, description="Whether to normalize query embedding")


class SearchResult(BaseModel):
    id: int
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="Top k search results")
    query_text: str = Field(..., description="Original query text")
    k: int = Field(..., description="Number of results requested")


class VectorStoreStatsResponse(BaseModel):
    total_vectors: int
    dimension: int
    initialized: bool


class SimilarityRequest(BaseModel):
    text1: str = Field(..., min_length=1, description="First text to compare")
    text2: str = Field(..., min_length=1, description="Second text to compare")
    normalize: bool = Field(True, description="Whether to normalize embeddings")


class SimilarityResponse(BaseModel):
    text1: str = Field(..., description="First text")
    text2: str = Field(..., description="Second text")
    similarity: float = Field(..., ge=-1.0, le=1.0, description="Similarity score (cosine similarity, range -1 to 1)")
    dimension: int = Field(..., description="Embedding dimension")


@app.on_event("startup")
async def startup_event():
    """Load model and initialize vector store on startup."""
    logger.info("Starting BGE-M3 Embedding Service...")
    try:
        embedding_service.load_model()
        vector_store.initialize()
        logger.info("Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Save vector store on shutdown."""
    try:
        vector_store.save()
        logger.info("Vector store saved successfully")
    except Exception as e:
        logger.error(f"Failed to save vector store: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        is_loaded = embedding_service.is_model_loaded()
        return HealthResponse(
            status="healthy" if is_loaded else "unhealthy",
            model_loaded=is_loaded,
            model_name=settings.model_name,
            dimension=settings.embedding_dim
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info", response_model=InfoResponse)
async def get_info():
    """Get model information."""
    try:
        return InfoResponse(
            model_name=settings.model_name,
            dimension=settings.embedding_dim,
            description="BGE-M3 embedding model for text encoding"
        )
    except Exception as e:
        logger.error(f"Failed to get info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode", response_model=EmbeddingResponse)
async def encode_texts(request: EncodeRequest):
    """Encode multiple texts to embeddings."""
    try:
        if not embedding_service.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        embeddings = embedding_service.encode(
            texts=request.texts,
            normalize=request.normalize
        )
        
        # Convert numpy array to list of lists
        embeddings_list = embeddings.tolist()
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimension=settings.embedding_dim,
            count=len(embeddings_list)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to encode texts: {e}")
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")


@app.post("/encode/single", response_model=SingleEmbeddingResponse)
async def encode_single(request: EncodeSingleRequest):
    """Encode a single text to embedding."""
    try:
        if not embedding_service.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        embedding = embedding_service.encode_single(
            text=request.text,
            normalize=request.normalize
        )
        
        # Convert numpy array to list
        embedding_list = embedding.tolist()
        
        return SingleEmbeddingResponse(
            embedding=embedding_list,
            dimension=settings.embedding_dim
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to encode text: {e}")
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")


@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """Calculate similarity score between two texts."""
    try:
        if not embedding_service.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Encode both texts
        embeddings = embedding_service.encode(
            texts=[request.text1, request.text2],
            normalize=request.normalize
        )
        
        if len(embeddings) != 2:
            raise HTTPException(status_code=500, detail="Failed to encode texts")
        
        # Calculate cosine similarity (dot product for normalized vectors)
        embedding1 = embeddings[0]
        embedding2 = embeddings[1]
        
        # Cosine similarity = dot product of normalized vectors
        similarity = float(np.dot(embedding1, embedding2))
        
        return SimilarityResponse(
            text1=request.text1,
            text2=request.text2,
            similarity=similarity,
            dimension=settings.embedding_dim
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate similarity: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")


@app.post("/store", response_model=StoreResponse)
async def store_texts(request: StoreRequest):
    """Embed texts and store them in vector database."""
    try:
        if not embedding_service.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Encode texts
        embeddings = embedding_service.encode(
            texts=request.texts,
            normalize=request.normalize
        )
        
        # Store in vector database
        ids = vector_store.add_vectors(
            embeddings=embeddings,
            texts=request.texts,
            metadata=request.metadata
        )
        
        # Auto-save after adding
        vector_store.save()
        
        return StoreResponse(
            ids=ids,
            count=len(ids),
            total_vectors=vector_store.ntotal
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to store texts: {e}")
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_texts(request: SearchRequest):
    """Search for top k similar texts in vector database."""
    try:
        if not embedding_service.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if not vector_store.is_initialized() or vector_store.ntotal == 0:
            raise HTTPException(status_code=404, detail="Vector store is empty")
        
        # Encode query text
        query_embedding = embedding_service.encode_single(
            text=request.text,
            normalize=request.normalize
        )
        
        # Search in vector store
        distances, indices, metadata_list = vector_store.search(
            query_embedding=query_embedding,
            k=request.k
        )
        
        # Build results
        results = []
        for i, (distance, idx, meta) in enumerate(zip(distances, indices, metadata_list)):
            result = SearchResult(
                id=int(idx),
                text=meta.get("text", ""),
                score=float(distance),
                metadata={k: v for k, v in meta.items() if k not in ["id", "text"]}
            )
            results.append(result)
        
        return SearchResponse(
            results=results,
            query_text=request.text,
            k=len(results)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search texts: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/store/stats", response_model=VectorStoreStatsResponse)
async def get_store_stats():
    """Get vector store statistics."""
    try:
        return VectorStoreStatsResponse(
            total_vectors=vector_store.ntotal,
            dimension=vector_store.dimension,
            initialized=vector_store.is_initialized()
        )
    except Exception as e:
        logger.error(f"Failed to get store stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/store/clear")
async def clear_store():
    """Clear all vectors from the store."""
    try:
        vector_store.clear()
        vector_store.save()
        return {"message": "Vector store cleared successfully", "total_vectors": 0}
    except Exception as e:
        logger.error(f"Failed to clear store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "BGE-M3 Embedding Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "encode": "/encode",
            "encode_single": "/encode/single",
            "similarity": "/similarity",
            "store": "/store",
            "search": "/search",
            "store_stats": "/store/stats",
            "store_clear": "/store/clear",
            "docs": "/docs"
        }
    }


