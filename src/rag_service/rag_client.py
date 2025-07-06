# ==========================================
# FILE: src/rag_service/rag_client.py
# ==========================================
"""
Lightweight RAG client for Streamlit app.
Connects to the background RAG service instead of loading models directly.
"""

import requests
import streamlit as st
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RAGServiceClient:
    """Client to connect to the background RAG service."""
    
    def __init__(self, service_url: str = "http://127.0.0.1:8001"):
        self.service_url = service_url.rstrip('/')
        self._is_available = None
        self._error_message = None
    
    def check_service_health(self) -> tuple[bool, str]:
        """Check if RAG service is available."""
        if self._is_available is not None:
            return self._is_available, self._error_message
        
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("is_initialized"):
                    self._is_available = True
                    self._error_message = None
                else:
                    self._is_available = False
                    self._error_message = health_data.get("error", "Service not initialized")
            else:
                self._is_available = False
                self._error_message = f"Service returned status {response.status_code}"
        except requests.exceptions.RequestException as e:
            self._is_available = False
            self._error_message = f"Cannot connect to RAG service: {e}"
        
        return self._is_available, self._error_message
    
    def semantic_search(self, query: str, k: int = 5, period_filter: str = None) -> List[Dict[str, Any]]:
        """Perform semantic search via the service."""
        is_available, error = self.check_service_health()
        if not is_available:
            raise Exception(f"RAG service not available: {error}")
        
        try:
            response = requests.post(
                f"{self.service_url}/semantic_search",
                json={
                    "query": query,
                    "k": k,
                    "period_filter": period_filter
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Semantic search request failed: {e}")
            raise Exception(f"Search failed: {e}")
    
    def ask_question(self, question: str, period_filter: str = None) -> Dict[str, Any]:
        """Ask a question via the service."""
        is_available, error = self.check_service_health()
        if not is_available:
            raise Exception(f"RAG service not available: {error}")
        
        try:
            response = requests.post(
                f"{self.service_url}/ask_question",
                json={
                    "question": question,
                    "period_filter": period_filter
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Question request failed: {e}")
            raise Exception(f"Question failed: {e}")
    
    def analyze_language_evolution(self, word: str, periods: List[str] = None) -> Dict[str, Any]:
        """Analyze language evolution via the service."""
        is_available, error = self.check_service_health()
        if not is_available:
            raise Exception(f"RAG service not available: {error}")
        
        try:
            response = requests.post(
                f"{self.service_url}/language_evolution",
                json={
                    "word": word,
                    "periods": periods
                },
                timeout=45
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Evolution analysis request failed: {e}")
            raise Exception(f"Evolution analysis failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics via the service."""
        is_available, error = self.check_service_health()
        if not is_available:
            raise Exception(f"RAG service not available: {error}")
        
        try:
            response = requests.get(f"{self.service_url}/statistics", timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Statistics request failed: {e}")
            raise Exception(f"Statistics failed: {e}")

@st.cache_resource
def get_rag_client():
    """Get cached RAG client instance."""
    return RAGServiceClient()

# Enhanced StreamlitRAGManager that uses the service
class StreamlitRAGManager:
    """Enhanced RAG manager that connects to background service."""
    
    def __init__(self):
        self.client = get_rag_client()
        self.is_initialized = None
        self.initialization_error = None
        
    def initialize_rag(self):
        """Check if RAG service is available."""
        if self.is_initialized is not None:
            return self.is_initialized
        
        is_available, error = self.client.check_service_health()
        self.is_initialized = is_available
        self.initialization_error = error
        
        return is_available
    
    def semantic_search(self, query: str, k: int = 5, period_filter: str = None):
        """Perform semantic search if service is available."""
        try:
            period = None if period_filter == "All Periods" else period_filter
            results = self.client.semantic_search(query, k=k, period_filter=period)
            return results
        except Exception as e:
            st.error(f"Semantic search failed: {e}")
            return []
    
    def ask_question(self, question: str, period_filter: str = None):
        """Ask a question if service is available."""
        try:
            period = None if period_filter == "All Periods" else period_filter
            result = self.client.ask_question(question, period_filter=period)
            return result
        except Exception as e:
            st.error(f"Question failed: {e}")
            return {"answer": "Service unavailable", "source_documents": []}
    
    def analyze_language_evolution(self, word: str, periods: List[str] = None):
        """Analyze language evolution if service is available."""
        try:
            result = self.client.analyze_language_evolution(word, periods)
            return result
        except Exception as e:
            st.error(f"Evolution analysis failed: {e}")
            return {"word": word, "periods": {}, "summary": "Service unavailable"}
    
    def get_statistics(self):
        """Get statistics if service is available."""
        try:
            return self.client.get_statistics()
        except Exception as e:
            st.error(f"Statistics failed: {e}")
            return {}