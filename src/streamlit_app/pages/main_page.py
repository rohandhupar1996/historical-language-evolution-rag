# ==========================================
# FILE: src/streamlit_app/pages/main_page.py
# ==========================================
"""Main page layout and logic."""

import streamlit as st
from ..components.header import render_header, render_statistics
from ..components.sidebar import render_sidebar, render_search_history
from ..components.search_interface import render_search_input
from ..components.results_display import render_search_results, render_evolution_results
from ..components.visualizations import render_visualization_dashboard
from ..components.research_tools import render_research_tools
from ..utils.session_manager import SessionManager
from ..utils.data_processing import DataProcessor

def render_main_page():
    """Render the main application page."""
    # Initialize session
    SessionManager.initialize_session()
    
    # Initialize RAG system (cached)
    if not SessionManager.get_rag_system():
        rag_system = DataProcessor.initialize_rag_system()
        SessionManager.set_rag_system(rag_system)
    
    # Render header
    render_header()
    render_statistics()
    
    # Render sidebar and get controls
    sidebar_data = render_sidebar()
    render_search_history()
    
    # Handle clear action
    if sidebar_data['clear_clicked']:
        SessionManager.clear_session()
        st.rerun()
    
    # Render search input
    query = render_search_input()
    
    # Handle search
    if sidebar_data['search_clicked'] and query.strip():
        with st.spinner("Searching historical corpus..."):
            period = sidebar_data['period_filter'] if sidebar_data['period_filter'] != "All Periods" else None
            genre = sidebar_data['genre_filter'] if sidebar_data['genre_filter'] != "All Genres" else None
            
            rag_system = SessionManager.get_rag_system()
            results = DataProcessor.perform_search(rag_system, query, period, genre, limit=10)
            
            SessionManager.set_search_results(results)
            SessionManager.add_to_search_history(query, len(results))
    
    # Handle evolution analysis
    if sidebar_data['evolution_clicked'] and query.strip():
        with st.spinner("Analyzing language evolution..."):
            rag_system = SessionManager.get_rag_system()
            evolution_data = DataProcessor.analyze_evolution(rag_system, query.split()[0])
            
            if evolution_data:
                render_evolution_results(evolution_data)
    
    # Display search results
    results = SessionManager.get_search_results()
    if results:
        render_search_results(results)
    
    # Render visualization dashboard
    render_visualization_dashboard()
    
    # Render research tools
    render_research_tools()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏛️ Historical German Language Evolution RAG System</p>
        <p>Built with Streamlit | <a href="https://github.com/[username]/historical-language-evolution-rag" target="_blank">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)