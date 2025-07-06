# ==========================================
# FILE: src/streamlit_app/components/sidebar.py
# ==========================================
"""Sidebar component for search controls."""

import streamlit as st
from ..config import AppConfig
from ..utils.session_manager import SessionManager

def render_sidebar():
    """Render the sidebar with search controls."""
    st.sidebar.header("🔍 Search Controls")
    
    # Example queries
    st.sidebar.subheader("Example Queries")
    selected_example = st.sidebar.selectbox(
        "Choose an example query:",
        [""] + AppConfig.EXAMPLE_QUERIES,
        key="example_selector"
    )
    
    if selected_example:
        SessionManager.update_current_query(selected_example)
    
    # Filters
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        period_filter = st.selectbox(
            "Period:",
            AppConfig.PERIODS,
            key="period_filter"
        )
    
    with col2:
        genre_filter = st.selectbox(
            "Genre:",
            AppConfig.GENRES,
            key="genre_filter"
        )
    
    # Search controls
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        search_clicked = st.button("🔍 Search", type="primary", key="search_btn")
    
    with col2:
        evolution_clicked = st.button("📈 Evolution", key="evolution_btn")
    
    with col3:
        clear_clicked = st.button("🗑️ Clear", key="clear_btn")
    
    return {
        'selected_example': selected_example,
        'period_filter': period_filter,
        'genre_filter': genre_filter,
        'search_clicked': search_clicked,
        'evolution_clicked': evolution_clicked,
        'clear_clicked': clear_clicked
    }

def render_search_history():
    """Render search history in sidebar."""
    history = SessionManager.get_search_history()
    
    if history:
        st.sidebar.header("📝 Search History")
        for i, search in enumerate(history[-5:]):  # Show last 5
            if st.sidebar.button(
                f"{search['query'][:30]}... ({search['results_count']} results)",
                key=f"history_{i}"
            ):
                SessionManager.update_current_query(search['query'])
                st.rerun()
