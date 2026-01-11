"""BGE-M3 Embedding Service - FastAPI application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import logging

from .service import EmbeddingService
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


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting BGE-M3 Embedding Service...")
    try:
        embedding_service.load_model()
        logger.info("Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise


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
            "docs": "/docs"
        }
    }


