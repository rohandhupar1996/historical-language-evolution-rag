# ==========================================
# FILE: src/streamlit_app/components/research_tools.py
# ==========================================
"""Research tools component."""

import streamlit as st
import json
from datetime import datetime
from typing import List, Dict, Any
from ..utils.export_utils import ExportUtils
from ..utils.session_manager import SessionManager

def render_research_tools():
    """Render research tools section."""
    st.header("🔬 Research Tools")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_export_citations()
    
    with col2:
        render_download_data()
    
    with col3:
        render_share_findings()
    
    with col4:
        render_generate_report()

def render_export_citations():
    """Render export citations tool."""
    if st.button("📚 Export Citations"):
        results = SessionManager.get_search_results()
        if results:
            citations = ExportUtils.generate_bibtex_citations(results)
            st.download_button(
                "Download BibTeX",
                citations,
                file_name="citations.bib",
                mime="text/plain"
            )
        else:
            st.warning("No results to export. Please perform a search first.")

def render_download_data():
    """Render download data tool."""
    if st.button("💾 Download Data"):
        results = SessionManager.get_search_results()
        query = SessionManager.get_current_query()
        
        if results:
            data = ExportUtils.prepare_json_export(query, results)
            st.download_button(
                "Download JSON",
                data,
                file_name="rag_search_results.json",
                mime="application/json"
            )
        else:
            st.warning("No data to download. Please perform a search first.")

def render_share_findings():
    """Render share findings tool."""
    if st.button("🔗 Share Findings"):
        query = SessionManager.get_current_query()
        if query:
            share_url = ExportUtils.generate_share_url(query)
            st.code(share_url)
            st.success("Shareable URL generated!")
        else:
            st.warning("Enter a query to generate shareable link.")

def render_generate_report():
    """Render generate report tool."""
    if st.button("📄 Generate Report"):
        results = SessionManager.get_search_results()
        query = SessionManager.get_current_query()
        
        if results:
            report = ExportUtils.generate_markdown_report(query, results)
            st.download_button(
                "Download Report",
                report,
                file_name="research_report.md",
                mime="text/markdown"
            )
        else:
            st.warning("No results to report. Please perform a search first.")