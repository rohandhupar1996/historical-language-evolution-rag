# ==========================================
# FILE: streamlit_app.py (Main entry point)
# ==========================================
"""Main Streamlit application entry point."""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.streamlit_app.config import AppConfig
from src.streamlit_app.pages.main_page import render_main_page
from src.streamlit_app.pages.evolution_analysis import render_evolution_analysis_page
from src.streamlit_app.pages.advanced_search import render_advanced_search_page

def main():
    """Main application function."""
    # Configure page
    st.set_page_config(
        page_title=AppConfig.APP_TITLE,
        page_icon=AppConfig.APP_ICON,
        layout=AppConfig.LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    # Navigation
    pages = {
        "🏠 Main Search": render_main_page,
        "📈 Evolution Analysis": render_evolution_analysis_page,
        "🔍 Advanced Search": render_advanced_search_page
    }
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.selectbox("Choose a page:", list(pages.keys()))
    
    # Render selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()