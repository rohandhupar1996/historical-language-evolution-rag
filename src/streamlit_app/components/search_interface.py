# ==========================================
# FILE: src/streamlit_app/components/search_interface.py
# ==========================================
"""Search interface component."""

import streamlit as st
from ..utils.session_manager import SessionManager

def render_search_input():
    """Render the main search input area."""
    current_query = SessionManager.get_current_query()
    
    query = st.text_area(
        "Enter your research question:",
        value=current_query,
        height=100,
        placeholder="Enter your research question in German or English...",
        key="main_query_input"
    )
    
    return query

def render_search_status():
    """Render search status and loading indicators."""
    if 'searching' in st.session_state and st.session_state.searching:
        with st.spinner("Searching historical corpus..."):
            st.empty()