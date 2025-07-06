# ==========================================
# FILE: src/streamlit_app/pages/advanced_search.py
# ==========================================
"""Advanced search page."""

import streamlit as st
from ..utils.session_manager import SessionManager
from ..utils.data_processing import DataProcessor
from ..components.results_display import render_search_results

def render_advanced_search_page():
    """Render advanced search page."""
    st.title("🔍 Advanced Search")
    
    # Search form
    with st.form("advanced_search_form"):
        query = st.text_area("Search Query:", height=100)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            period = st.selectbox("Period:", ["All"] + [p for p in [
                "1050-1350", "1350-1650", "1650-1700", 
                "1700-1800", "1800-1900", "1900-2000"
            ]])
        
        with col2:
            genre = st.selectbox("Genre:", ["All"] + [
                "Drama", "Humanities", "Legal", 
                "Narrative", "Newspapers", "Scientific"
            ])
        
        with col3:
            limit = st.slider("Max Results:", 5, 50, 10)
        
        submitted = st.form_submit_button("🔍 Search", type="primary")
    
    if submitted and query.strip():
        rag_system = SessionManager.get_rag_system()
        
        period_filter = period if period != "All" else None
        genre_filter = genre if genre != "All" else None
        
        with st.spinner("Searching..."):
            results = DataProcessor.perform_search(
                rag_system, query, period_filter, genre_filter, limit
            )
            
            SessionManager.set_search_results(results)
            render_search_results(results)