# ==========================================
# FILE: src/streamlit_app/components/visualizations.py
# ==========================================
"""Visualization components."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def render_visualization_dashboard():
    """Render the complete visualization dashboard."""
    st.header("📊 Visualization Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Timeline", 
        "☁️ Word Frequency", 
        "📚 Genre Distribution", 
        "🔄 Evolution"
    ])
    
    with tab1:
        render_timeline_chart()
    
    with tab2:
        render_word_frequency_chart()
    
    with tab3:
        render_genre_chart()
    
    with tab4:
        render_evolution_visualization()

def render_timeline_chart():
    """Render timeline chart."""
    periods = ['1050-1350', '1350-1650', '1650-1700', '1700-1800', '1800-1900', '1900-2000']
    chunks = [245, 678, 902, 1156, 534, 92]
    
    fig = px.line(
        x=periods, 
        y=chunks,
        title="Temporal Distribution of Text Chunks",
        labels={'x': 'Time Period', 'y': 'Number of Chunks'}
    )
    fig.update_traces(line_color='#2a5298', line_width=3)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Temporal distribution of text chunks across historical periods")

def render_word_frequency_chart():
    """Render word frequency chart."""
    words = ['und', 'der', 'die', 'von', 'zu', 'mit', 'in', 'ist', 'das', 'sich']
    frequencies = [12430, 8765, 7432, 6543, 5678, 4321, 3987, 3456, 3210, 2987]
    
    fig = px.bar(
        x=words,
        y=frequencies,
        title="Most Frequent Words in Corpus",
        labels={'x': 'Words', 'y': 'Frequency'}
    )
    fig.update_traces(marker_color='#2a5298')
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Most frequent terms in the corpus")

def render_genre_chart():
    """Render genre distribution chart."""
    genres = ['Drama', 'Humanities', 'Legal', 'Narrative', 'Newspapers', 'Scientific']
    counts = [456, 789, 902, 1234, 567, 678]
    
    fig = px.pie(
        values=counts,
        names=genres,
        title="Genre Distribution in Corpus"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Distribution of text genres in the corpus")

def render_evolution_visualization():
    """Render evolution visualization."""
    from ..utils.session_manager import SessionManager
    
    results = SessionManager.get_search_results()
    if results:
        st.write("### Evolution Pattern")
        for result in results[:3]:
            st.write(f"**{result['metadata']['period']}** - {result['metadata']['genre']}")
            st.write(f"*{result['text'][:100]}...*")
            st.write("---")
    else:
        st.info("Perform a search to see evolution patterns")