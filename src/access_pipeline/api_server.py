# ==========================================
# FILE: access_pipeline/api_server.py
# ==========================================
"""FastAPI server implementation."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Optional
from .models import QueryRequest, EvolutionQuery, SearchRequest
from .query_handlers import QueryHandlers
from .config import API_CONFIG


class GerManCAPI:
    """FastAPI application for GerManC corpus."""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.query_handlers = QueryHandlers(db_config)
        self.app = FastAPI(
            title=API_CONFIG['title'],
            version=API_CONFIG['version']
        )
        self._setup_cors()
        self._setup_routes()
    
    def _setup_cors(self):
        """Setup CORS middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        def root():
            return {"message": API_CONFIG['title'], "version": API_CONFIG['version']}
        
        @self.app.get("/evolution/{word}/{start_period}/{end_period}")
        def query_word_evolution(word: str, start_period: str, end_period: str):
            """Track word evolution across periods."""
            try:
                return self.query_handlers.query_word_evolution(word, start_period, end_period)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/linguistic_analysis")
        def linguistic_analysis(request: QueryRequest):
            """Perform linguistic pattern analysis."""
            try:
                return self.query_handlers.linguistic_analysis(request)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/temporal_patterns")
        def temporal_patterns():
            """Get temporal distribution patterns."""
            try:
                return self.query_handlers.get_temporal_patterns()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/search/{query}")
        def full_text_search(query: str, period: Optional[str] = None, limit: int = 50):
            """Full-text search in historical texts."""
            try:
                return self.query_handlers.full_text_search(query, period, limit)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = None, port: int = None):
        """Start API server."""
        host = host or API_CONFIG['host']
        port = port or API_CONFIG['port']
        
        print(f"🌐 Server running at: http://{host}:{port}")
        print(f"📖 API docs at: http://{host}:{port}/docs")
        
        uvicorn.run(self.app, host=host, port=port)