# ==========================================
# FILE: src/streamlit_app/utils/session_manager.py
# ==========================================
"""Session state management utilities."""

import streamlit as st
from typing import List, Dict, Any
from datetime import datetime

class SessionManager:
    """Manages Streamlit session state."""
    
    @staticmethod
    def initialize_session():
        """Initialize session state variables."""
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        if 'current_query' not in st.session_state:
            st.session_state.current_query = ""
        
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
    
    @staticmethod
    def get_search_results() -> List[Dict[str, Any]]:
        """Get current search results."""
        return st.session_state.get('search_results', [])
    
    @staticmethod
    def set_search_results(results: List[Dict[str, Any]]):
        """Set search results."""
        st.session_state.search_results = results
    
    @staticmethod
    def get_current_query() -> str:
        """Get current query."""
        return st.session_state.get('current_query', "")
    
    @staticmethod
    def update_current_query(query: str):
        """Update current query."""
        st.session_state.current_query = query
    
    @staticmethod
    def add_to_search_history(query: str, results_count: int):
        """Add search to history."""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        st.session_state.search_history.append({
            'query': query,
            'timestamp': datetime.now(),
            'results_count': results_count
        })
    
    @staticmethod
    def get_search_history() -> List[Dict[str, Any]]:
        """Get search history."""
        return st.session_state.get('search_history', [])
    
    @staticmethod
    def clear_session():
        """Clear all session data."""
        st.session_state.search_results = []
        st.session_state.current_query = ""
    
    @staticmethod
    def get_rag_system():
        """Get RAG system instance."""
        return st.session_state.get('rag_system')
    
    @staticmethod
    def set_rag_system(rag_system):
        """Set RAG system instance."""
        st.session_state.rag_system = rag_system
