"""Configuration for embedding service."""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model configuration
    model_name: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 3004
    
    # Logging configuration
    log_level: str = "INFO"
    
    # Vector store configuration
    vector_store_path: Optional[str] = None  # None means use default path
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


