# ==========================================
# FILE: src/streamlit_app/components/header.py
# ==========================================
"""Header component for the Streamlit app."""

import streamlit as st
from ..config import AppConfig
from ..styles.custom_css import get_custom_css

def render_header():
    """Render the main header with title and description."""
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ German Historical Corpus RAG System</h1>
        <p>Explore the evolution of German language from 1050-2000 through advanced semantic search and AI analysis</p>
    </div>
    """, unsafe_allow_html=True)

def render_statistics():
    """Render statistics dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    stats = AppConfig.SAMPLE_STATS
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{stats['total_chunks']:,}</h2>
            <p>Text Chunks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{stats['total_periods']}</h2>
            <p>Time Periods</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{stats['total_genres']}</h2>
            <p>Text Genres</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{stats['total_embeddings']}</h2>
            <p>Embeddings Created</p>
        </div>
        """, unsafe_allow_html=True)
