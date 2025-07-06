# ==========================================
# FILE: src/rag_service/rag_server.py
# ==========================================
"""
Persistent RAG service that runs in the background.
This keeps models loaded and provides API endpoints for the Streamlit app.
"""

import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag_system.pipeline import GermanRAGPipeline
from rag_system.config import DEFAULT_DB_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """Persistent RAG service with loaded models."""
    
    def __init__(self):
        self.rag_pipeline = None
        self.is_initialized = False
        self.initialization_error = None
        
    async def initialize(self):
        """Initialize RAG pipeline once at startup."""
        if self.is_initialized:
            return
            
        try:
            logger.info("🚀 Initializing RAG pipeline...")
            
            # Database config
            db_config = DEFAULT_DB_CONFIG.copy()
            db_config.update({
                'host': 'localhost',
                'port': 5432,
                'database': 'germanc_corpus',
                'user': 'rohan',
                'password': '1996'
            })
            
            # Find vector database path
            import os
            vector_db_path = os.getenv('VECTOR_DB_PATH')
            
            if not vector_db_path:
                possible_paths = [
                    "./german_corpus_vectordb",
                    "/Users/rohan/Downloads/historical-language-evolution-rag/german_corpus_vectordb",
                    "../german_corpus_vectordb",
                    os.path.expanduser("~/Downloads/historical-language-evolution-rag/german_corpus_vectordb")
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        vector_db_path = path
                        logger.info(f"📦 Found vector database at: {path}")
                        break
            else:
                logger.info(f"📦 Using vector database from environment: {vector_db_path}")
            
            if not vector_db_path or not Path(vector_db_path).exists():
                raise Exception(f"Vector database not found. Checked: {possible_paths if not vector_db_path else [vector_db_path]}")
            
            # Initialize pipeline
            self.rag_pipeline = GermanRAGPipeline(db_config, vector_db_path)
            
            # Setup QA system
            self.rag_pipeline.setup_qa_system("simple")
            
            self.is_initialized = True
            logger.info("✅ RAG pipeline initialized successfully!")
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ RAG initialization failed: {e}")
            self.is_initialized = False

# Global RAG service instance
rag_service = RAGService()

# FastAPI app
app = FastAPI(
    title="GerManC RAG Service",
    description="Background service for AI-powered semantic search",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SemanticSearchRequest(BaseModel):
    query: str
    k: int = 5
    period_filter: Optional[str] = None

class SemanticSearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int

class QuestionRequest(BaseModel):
    question: str
    period_filter: Optional[str] = None

class EvolutionRequest(BaseModel):
    word: str
    periods: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    is_initialized: bool
    error: Optional[str] = None

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize RAG service on startup."""
    await rag_service.initialize()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if rag_service.is_initialized else "error",
        is_initialized=rag_service.is_initialized,
        error=rag_service.initialization_error
    )

@app.post("/semantic_search", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """Perform semantic search."""
    if not rag_service.is_initialized:
        raise HTTPException(
            status_code=503, 
            detail=f"RAG service not initialized: {rag_service.initialization_error}"
        )
    
    try:
        period = None if request.period_filter == "All Periods" else request.period_filter
        results = rag_service.rag_pipeline.semantic_search(
            request.query, 
            k=request.k, 
            period_filter=period
        )
        
        return SemanticSearchResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    """Ask a question using RAG."""
    if not rag_service.is_initialized:
        raise HTTPException(
            status_code=503, 
            detail=f"RAG service not initialized: {rag_service.initialization_error}"
        )
    
    try:
        period = None if request.period_filter == "All Periods" else request.period_filter
        result = rag_service.rag_pipeline.ask_question(
            request.question,
            period_filter=period
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/language_evolution")
async def language_evolution(request: EvolutionRequest):
    """Analyze language evolution."""
    if not rag_service.is_initialized:
        raise HTTPException(
            status_code=503, 
            detail=f"RAG service not initialized: {rag_service.initialization_error}"
        )
    
    try:
        result = rag_service.rag_pipeline.analyze_language_evolution(
            request.word,
            periods=request.periods
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Language evolution analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get system statistics."""
    if not rag_service.is_initialized:
        raise HTTPException(
            status_code=503, 
            detail=f"RAG service not initialized: {rag_service.initialization_error}"
        )
    
    try:
        return rag_service.rag_pipeline.get_statistics()
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_rag_service(host: str = "127.0.0.1", port: int = 8001):
    """Run the RAG service."""
    print("🚀 Starting GerManC RAG Service...")
    print(f"📡 Service will be available at: http://{host}:{port}")
    print("📚 API docs at: http://{host}:{port}/docs")
    print("💡 Keep this service running while using Streamlit app!")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    run_rag_service()