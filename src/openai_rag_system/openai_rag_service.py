# ==========================================
# FILE: src/openai_rag_system/openai_rag_service.py
# ==========================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import uvicorn

from .pipeline import OpenAIGermanRAGPipeline
from .config import DEFAULT_DB_CONFIG

# Request models
class SemanticSearchRequest(BaseModel):
    query: str
    k: int = 5
    period_filter: Optional[str] = None

class QuestionRequest(BaseModel):
    question: str
    period_filter: Optional[str] = None
    analysis_depth: str = "standard"

class EvolutionRequest(BaseModel):
    word_or_concept: str
    periods: Optional[List[str]] = None
    analysis_type: str = "comprehensive"

class InsightsRequest(BaseModel):
    topic: str
    insight_type: str = "linguistic_evolution"

# Global service instance
openai_rag_service = None

app = FastAPI(
    title="OpenAI German Historical Linguistics RAG",
    description="Advanced AI-powered analysis of German language evolution",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global openai_rag_service
    try:
        print("🤖 Initializing OpenAI RAG Pipeline...")
        
        db_config = DEFAULT_DB_CONFIG.copy()
        vector_db_path = "./openai_vector_db"
        
        openai_rag_service = OpenAIGermanRAGPipeline(db_config, vector_db_path)
        
        # Load corpus and create embeddings
        chunks_df = openai_rag_service.load_and_embed_corpus(limit=1000)  # Limit for demo
        
        # Setup QA system
        openai_rag_service.setup_advanced_qa_system()
        
        print("✅ OpenAI RAG Service ready!")
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        openai_rag_service = None

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if openai_rag_service else "error",
        "is_initialized": openai_rag_service is not None,
        "service_type": "OpenAI RAG"
    }

@app.post("/semantic_search")
async def semantic_search(request: SemanticSearchRequest):
    if not openai_rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        results = openai_rag_service.semantic_search(
            request.query, 
            k=request.k, 
            period_filter=request.period_filter
        )
        
        return {
            "query": request.query,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    if not openai_rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        result = openai_rag_service.ask_question(
            request.question,
            period_filter=request.period_filter,
            analysis_depth=request.analysis_depth
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/language_evolution")
async def language_evolution(request: EvolutionRequest):
    if not openai_rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        result = openai_rag_service.analyze_language_evolution(
            request.word_or_concept,
            periods=request.periods,
            analysis_type=request.analysis_type
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/historical_insights")
async def historical_insights(request: InsightsRequest):
    if not openai_rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        result = openai_rag_service.generate_historical_insights(
            request.topic,
            insight_type=request.insight_type
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    if not openai_rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return openai_rag_service.get_advanced_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_tests")
async def run_tests():
    if not openai_rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return openai_rag_service.run_comprehensive_tests()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_openai_rag_service(host: str = "127.0.0.1", port: int = 8002):
    print("🤖 Starting OpenAI RAG Service...")
    print(f"📡 Service will be available at: http://{host}:{port}")
    print("🧠 Using OpenAI GPT-4 and text-embedding-3-large")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_openai_rag_service()