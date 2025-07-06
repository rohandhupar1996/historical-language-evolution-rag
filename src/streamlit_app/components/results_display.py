# ==========================================
# FILE: src/streamlit_app/components/results_display.py
# ==========================================
"""Results display component."""

import streamlit as st
from typing import List, Dict, Any
import random
from ..config import AppConfig

def render_linguistic_insight(results_count: int):
    """Render linguistic insight based on results."""
    insights = AppConfig.LINGUISTIC_INSIGHTS
    insight = insights[results_count % len(insights)]
    
    st.markdown(f"""
    <div class="insight-card">
        <strong>💡 Linguistic Insight:</strong> {insight}
    </div>
    """, unsafe_allow_html=True)

def render_search_results(results: List[Dict[str, Any]]):
    """Render search results."""
    if not results:
        st.info("No results found. Try a different query or adjust filters.")
        return
    
    st.header("📋 Search Results")
    
    # Show linguistic insight
    render_linguistic_insight(len(results))
    
    # Display results
    for i, result in enumerate(results):
        render_single_result(result, i)

def render_single_result(result: Dict[str, Any], index: int):
    """Render a single search result."""
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; gap: 15px; margin-bottom: 10px; font-size: 0.9rem; color: #666;">
            <span>📅 {result['metadata']['period']}</span>
            <span>📚 {result['metadata']['genre']}</span>
            <span>📄 {result['metadata']['filename']}</span>
        </div>
        <div style="margin-bottom: 10px; line-height: 1.6;">
            {result['text']}
        </div>
        <div style="font-size: 0.8rem; color: #888;">
            Confidence: {result['confidence'] * 100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button(f"🔍 Analyze Context", key=f"analyze_{index}"):
        render_context_analysis(result)

def render_context_analysis(result: Dict[str, Any]):
    """Render detailed context analysis."""
    st.info(f"""
    **Context Analysis for {result['metadata']['filename']}:**
    
    - **Period:** {result['metadata']['period']}
    - **Genre:** {result['metadata']['genre']}
    - **Text Length:** {len(result['text'])} characters
    - **Confidence:** {result['confidence'] * 100:.1f}%
    
    This text represents {result['metadata']['genre']} literature from the 
    {result['metadata']['period']} period, showing characteristic linguistic patterns 
    of Early New High German.
    """)

def render_evolution_results(evolution_data: Dict[str, Any]):
    """Render language evolution analysis results."""
    if not evolution_data:
        st.error("Evolution analysis failed.")
        return
    
    st.header("📈 Language Evolution Analysis")
    st.subheader(f"Evolution of '{evolution_data['word']}'")
    
    for period, data in evolution_data['periods'].items():
        with st.expander(f"{period} - {data['context_count']} contexts found"):
            for example in data['examples'][:3]:  # Show first 3 examples
                st.write(f"💬 {example['text']}")
    
    st.info(f"**Summary:** {evolution_data['summary']}")
