# ==========================================
# FILE: src/streamlit_app/utils/chart_utils.py
# ==========================================
"""Chart and visualization utilities."""

import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

class ChartUtils:
    """Utilities for creating charts and visualizations."""
    
    @staticmethod
    def create_timeline_chart(periods: List[str], chunks: List[int]):
        """Create timeline chart."""
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
        return fig
    
    @staticmethod
    def create_word_frequency_chart(words: List[str], frequencies: List[int]):
        """Create word frequency bar chart."""
        fig = px.bar(
            x=words,
            y=frequencies,
            title="Most Frequent Words in Corpus",
            labels={'x': 'Words', 'y': 'Frequency'}
        )
        fig.update_traces(marker_color='#2a5298')
        return fig
    
    @staticmethod
    def create_genre_pie_chart(genres: List[str], counts: List[int]):
        """Create genre distribution pie chart."""
        fig = px.pie(
            values=counts,
            names=genres,
            title="Genre Distribution in Corpus"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    @staticmethod
    def create_evolution_chart(evolution_data: Dict[str, Any]):
        """Create language evolution visualization."""
        periods = list(evolution_data['periods'].keys())
        counts = [data['context_count'] for data in evolution_data['periods'].values()]
        
        fig = px.bar(
            x=periods,
            y=counts,
            title=f"Evolution of '{evolution_data['word']}'",
            labels={'x': 'Time Period', 'y': 'Context Count'}
        )
        fig.update_traces(marker_color='#2a5298')
        return fig
