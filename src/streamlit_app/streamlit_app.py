# ==========================================
# FILE: Corrected Streamlit App - All Original Features + RAG
# ==========================================
"""
Complete Streamlit app preserving ALL original SQL analytics features
+ adding RAG capabilities as enhancements
"""

import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import sys
from collections import defaultdict

# Add src to path for RAG imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Page configuration
st.set_page_config(
    page_title="🏛️ German Historical Corpus Analytics + AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS (keeping all original styling + adding RAG)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.3);
    }
    
    .analytics-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #2a5298;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .result-card {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin-bottom: 1rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(42, 82, 152, 0.3);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .search-highlight {
        background-color: #fff3cd;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    
    .insights-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    
    .word-freq-item {
        background: #f0f8ff;
        padding: 0.5rem;
        margin: 0.2rem;
        border-radius: 5px;
        border-left: 3px solid #2a5298;
    }
    
    /* NEW RAG-specific styles */
    .rag-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .semantic-result {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #00b894;
    }
    
    .rag-status-ready {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .rag-status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'germanc_corpus',
    'user': 'rohan',
    'password': '1996'
}

def get_database_connection():
    """Get database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# ==============================================
# RAG SYSTEM INTEGRATION (NEW - OPTIONAL)
# ==============================================

class StreamlitRAGManager:
    """Optional RAG manager - doesn't interfere with existing features."""
    
    def __init__(self):
        self.rag_system = None
        self.is_initialized = False
        self.initialization_error = None
        
    def initialize_rag(self):
        """Initialize RAG system if available."""
        if self.is_initialized:
            return self.rag_system
        
        try:
            from rag_system.pipeline import GermanRAGPipeline
            from rag_system.config import DEFAULT_DB_CONFIG
            
            db_config = DEFAULT_DB_CONFIG.copy()
            db_config.update(DB_CONFIG)
            
            self.rag_system = GermanRAGPipeline(db_config, "./german_corpus_vectordb")
            
            vector_db_path = Path("./german_corpus_vectordb")
            if not vector_db_path.exists():
                return None
            
            self.is_initialized = True
            return self.rag_system
            
        except Exception as e:
            self.initialization_error = str(e)
            return None
    
    def semantic_search(self, query: str, k: int = 5, period_filter: str = None):
        """Perform semantic search if RAG is available."""
        if not self.is_initialized:
            self.initialize_rag()
        
        if not self.rag_system:
            return []
        
        try:
            period = None if period_filter == "All Periods" else period_filter
            results = self.rag_system.semantic_search(query, k=k, period_filter=period)
            return results
        except Exception as e:
            return []

@st.cache_resource
def get_rag_manager():
    """Get cached RAG manager instance."""
    return StreamlitRAGManager()

# ==============================================
# ALL ORIGINAL SQL ANALYTICS FUNCTIONS (PRESERVED)
# ==============================================

def word_frequency_over_time(word):
    """Track word usage changes over time periods."""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        query = """
        SELECT 
            period, 
            COUNT(*) as frequency,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage,
            STRING_AGG(DISTINCT genre, ', ') as genres_found
        FROM chunks 
        WHERE normalized_text ILIKE %s
        GROUP BY period 
        ORDER BY period
        """
        df = pd.read_sql(query, conn, params=[f'%{word}%'])
        return df
    except Exception as e:
        st.error(f"Error in word frequency analysis: {e}")
        return None
    finally:
        conn.close()

def genre_vocabulary_analysis():
    """Compare vocabulary richness across genres."""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        query = """
        WITH word_counts AS (
            SELECT 
                genre,
                COUNT(*) as total_chunks,
                SUM(token_count) as total_tokens,
                AVG(token_count) as avg_tokens_per_chunk,
                COUNT(DISTINCT doc_id) as unique_documents
            FROM chunks 
            GROUP BY genre
        )
        SELECT 
            genre,
            total_chunks,
            total_tokens,
            ROUND(avg_tokens_per_chunk::numeric, 1) as avg_tokens_per_chunk,
            unique_documents,
            ROUND((total_tokens::numeric / total_chunks::numeric), 1) as vocabulary_density
        FROM word_counts
        ORDER BY total_tokens DESC
        """
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error in genre analysis: {e}")
        return None
    finally:
        conn.close()

def co_occurrence_analysis(word1, word2):
    """Find co-occurrence patterns between two words."""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        query = """
        WITH word1_docs AS (
            SELECT chunk_id, period, genre, year, normalized_text
            FROM chunks 
            WHERE normalized_text ILIKE %s
        ),
        word2_docs AS (
            SELECT chunk_id, period, genre, year, normalized_text
            FROM chunks 
            WHERE normalized_text ILIKE %s
        ),
        cooccurrence AS (
            SELECT w1.chunk_id, w1.period, w1.genre, w1.year
            FROM word1_docs w1
            INNER JOIN word2_docs w2 ON w1.chunk_id = w2.chunk_id
        )
        SELECT 
            period, 
            genre,
            COUNT(*) as cooccurrences,
            AVG(year) as avg_year
        FROM cooccurrence
        GROUP BY period, genre
        ORDER BY cooccurrences DESC
        """
        df = pd.read_sql(query, conn, params=[f'%{word1}%', f'%{word2}%'])
        return df
    except Exception as e:
        st.error(f"Error in co-occurrence analysis: {e}")
        return None
    finally:
        conn.close()

def historical_spelling_analysis():
    """Analyze archaic spelling patterns."""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        query = """
        SELECT 
            period,
            COUNT(*) as total_chunks,
            COUNT(*) FILTER (WHERE normalized_text ~ '.*th.*') as th_patterns,
            COUNT(*) FILTER (WHERE normalized_text ~ '.*ck$') as ck_endings,
            COUNT(*) FILTER (WHERE normalized_text ~ '^v[aeiou]') as v_beginnings,
            COUNT(*) FILTER (WHERE normalized_text ~ '.*ey.*') as ey_diphthongs,
            ROUND(
                (COUNT(*) FILTER (WHERE normalized_text ~ '.*th.*|.*ck$|^v[aeiou]|.*ey.*') * 100.0 / COUNT(*))::numeric, 
                2
            ) as archaic_percentage
        FROM chunks
        GROUP BY period
        ORDER BY period
        """
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error in spelling analysis: {e}")
        return None
    finally:
        conn.close()

def text_complexity_analysis():
    """Analyze text complexity across periods."""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        query = """
        SELECT 
            period,
            genre,
            COUNT(*) as chunk_count,
            AVG(LENGTH(normalized_text)) as avg_text_length,
            AVG(token_count) as avg_word_count,
            AVG(LENGTH(normalized_text)::float / NULLIF(token_count, 0)) as avg_word_length,
            MIN(year) as earliest_year,
            MAX(year) as latest_year
        FROM chunks
        GROUP BY period, genre
        ORDER BY period, avg_word_length DESC
        """
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error in complexity analysis: {e}")
        return None
    finally:
        conn.close()

def top_words_by_period():
    """Get most frequent words by period."""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        query = """
        WITH word_extraction AS (
            SELECT 
                period,
                regexp_split_to_table(lower(normalized_text), '\s+') as word
            FROM chunks
            WHERE LENGTH(normalized_text) > 0
        ),
        word_counts AS (
            SELECT 
                period, 
                word, 
                COUNT(*) as frequency
            FROM word_extraction
            WHERE LENGTH(word) > 3
            AND word !~ '^[0-9]+$'
            AND word NOT IN ('und', 'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'eines', 'sich', 'ist', 'hat', 'war', 'kann', 'wird', 'auch', 'noch', 'nur', 'aber', 'oder', 'wenn', 'dann', 'denn', 'dass', 'daß')
            GROUP BY period, word
        ),
        ranked_words AS (
            SELECT 
                period, 
                word, 
                frequency,
                ROW_NUMBER() OVER (PARTITION BY period ORDER BY frequency DESC) as rank
            FROM word_counts
        )
        SELECT period, word, frequency
        FROM ranked_words
        WHERE rank <= 15
        ORDER BY period, rank
        """
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error in word frequency analysis: {e}")
        return None
    finally:
        conn.close()

def get_corpus_statistics():
    """Get corpus statistics."""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        query = """
        SELECT 
            COUNT(*) as total_chunks,
            COUNT(DISTINCT period) as periods,
            COUNT(DISTINCT genre) as genres,
            MIN(year) as earliest_year,
            MAX(year) as latest_year,
            AVG(token_count) as avg_tokens,
            SUM(token_count) as total_tokens
        FROM chunks
        """
        df = pd.read_sql(query, conn)
        return df.iloc[0].to_dict()
    except Exception as e:
        st.error(f"Error fetching statistics: {e}")
        return None
    finally:
        conn.close()

def search_database(query: str, period_filter: str = None, genre_filter: str = None, limit: int = 10):
    """Search the database using PostgreSQL text search."""
    conn = get_database_connection()
    if not conn:
        return [], []
    
    try:
        search_terms = query.split()
        where_conditions = []
        params = []
        
        for i, term in enumerate(search_terms):
            where_conditions.append(f"normalized_text ILIKE %s")
            params.append(f"%{term}%")
        
        search_condition = " AND ".join(where_conditions)
        
        if period_filter and period_filter != "All Periods":
            search_condition += " AND period = %s"
            params.append(period_filter)
        
        if genre_filter and genre_filter != "All Genres":
            search_condition += " AND genre = %s"
            params.append(genre_filter)
        
        search_query = f"""
        SELECT 
            chunk_id,
            normalized_text,
            original_text,
            period,
            genre,
            year,
            filename,
            token_count,
            doc_id
        FROM chunks
        WHERE {search_condition}
        ORDER BY period, year, chunk_id
        LIMIT %s
        """
        
        params.append(limit)
        df = pd.read_sql(search_query, conn, params=params)
        
        return df.to_dict('records'), search_terms
    except Exception as e:
        st.error(f"Search error: {e}")
        return [], []
    finally:
        conn.close()

def get_available_periods():
    """Get available periods from database."""
    conn = get_database_connection()
    if not conn:
        return ["All Periods"]
    
    try:
        query = "SELECT DISTINCT period FROM chunks ORDER BY period"
        df = pd.read_sql(query, conn)
        periods = ["All Periods"] + df['period'].tolist()
        return periods
    except Exception:
        return ["All Periods"]
    finally:
        conn.close()

def get_available_genres():
    """Get available genres from database."""
    conn = get_database_connection()
    if not conn:
        return ["All Genres"]
    
    try:
        query = "SELECT DISTINCT genre FROM chunks ORDER BY genre"
        df = pd.read_sql(query, conn)
        genres = ["All Genres"] + df['genre'].tolist()
        return genres
    except Exception:
        return ["All Genres"]
    finally:
        conn.close()

def highlight_search_terms(text: str, search_terms: List[str]) -> str:
    """Highlight search terms in text."""
    highlighted_text = text
    for term in search_terms:
        if term.strip():
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<span class="search-highlight">{term}</span>', 
                highlighted_text
            )
    return highlighted_text

# ==============================================
# ALL ORIGINAL VISUALIZATION FUNCTIONS (PRESERVED)
# ==============================================

def create_word_timeline_chart(df, word):
    """Create an interactive timeline chart for word frequency."""
    if df is None or df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['period'],
        y=df['frequency'],
        mode='lines+markers',
        name='Frequency',
        line=dict(color='#2a5298', width=3),
        marker=dict(size=10, color='#2a5298'),
        hovertemplate='<b>Period:</b> %{x}<br><b>Frequency:</b> %{y}<br><b>Percentage:</b> %{customdata:.1f}%<extra></extra>',
        customdata=df['percentage']
    ))
    
    fig.update_layout(
        title=f"📈 Frequency of '{word}' Over Time",
        xaxis_title="Time Period",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig

def create_genre_comparison_chart(df):
    """Create genre vocabulary comparison chart."""
    if df is None or df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Tokens by Genre', 'Average Tokens per Chunk', 
                       'Total Chunks by Genre', 'Vocabulary Density'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=df['genre'], y=df['total_tokens'], name='Total Tokens',
               marker_color='#2a5298'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df['genre'], y=df['avg_tokens_per_chunk'], name='Avg Tokens/Chunk',
               marker_color='#667eea'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=df['genre'], y=df['total_chunks'], name='Total Chunks',
               marker_color='#764ba2'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df['genre'], y=df['vocabulary_density'], name='Vocabulary Density',
               marker_color='#f093fb'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, template="plotly_white")
    
    return fig

def create_cooccurrence_heatmap(df, word1, word2):
    """Create a heatmap for word co-occurrence."""
    if df is None or df.empty:
        return None
    
    pivot_df = df.pivot_table(values='cooccurrences', index='period', columns='genre', fill_value=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='<b>Period:</b> %{y}<br><b>Genre:</b> %{x}<br><b>Co-occurrences:</b> %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"🔗 Co-occurrence Heatmap: '{word1}' & '{word2}'",
        xaxis_title="Genre",
        yaxis_title="Period",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_spelling_evolution_chart(df):
    """Create spelling evolution chart."""
    if df is None or df.empty:
        return None
    
    fig = go.Figure()
    
    patterns = ['th_patterns', 'ck_endings', 'v_beginnings', 'ey_diphthongs']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    pattern_names = ['TH Patterns', 'CK Endings', 'V Beginnings', 'EY Diphthongs']
    
    for pattern, color, name in zip(patterns, colors, pattern_names):
        fig.add_trace(go.Scatter(
            x=df['period'],
            y=df[pattern],
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="📜 Historical Spelling Pattern Evolution",
        xaxis_title="Time Period",
        yaxis_title="Pattern Frequency",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_complexity_scatter(df):
    """Create text complexity scatter plot."""
    if df is None or df.empty:
        return None
    
    fig = px.scatter(
        df, 
        x='avg_word_count', 
        y='avg_word_length',
        size='chunk_count',
        color='period',
        hover_data=['genre', 'earliest_year', 'latest_year'],
        title="📊 Text Complexity Analysis: Word Count vs Word Length",
        labels={
            'avg_word_count': 'Average Words per Chunk',
            'avg_word_length': 'Average Word Length',
            'chunk_count': 'Number of Chunks'
        }
    )
    
    fig.update_layout(template="plotly_white", height=500)
    
    return fig

def render_word_frequency_grid(words_df, period):
    """Render word frequency as a grid."""
    if words_df is None or words_df.empty:
        return
    
    period_words = words_df[words_df['period'] == period]
    
    if period_words.empty:
        st.warning(f"No word data found for period {period}")
        return
    
    st.markdown(f"### 📝 Most Frequent Words in {period}")
    
    cols = st.columns(3)
    
    for i, (_, row) in enumerate(period_words.head(15).iterrows()):
        with cols[i % 3]:
            font_size = max(12, min(24, int(20 * (row['frequency'] / period_words['frequency'].max()))))
            st.markdown(f"""
            <div class="word-freq-item">
                <span style="font-size: {font_size}px; font-weight: bold;">{row['word']}</span>
                <br><small>{row['frequency']} occurrences</small>
            </div>
            """, unsafe_allow_html=True)

# ==============================================
# ALL ORIGINAL RENDERING FUNCTIONS (PRESERVED)
# ==============================================

def render_search_results(results: List[Dict], search_terms: List[str], query: str):
    """Render search results with highlighting."""
    if not results:
        st.info("No results found. Try different search terms or adjust filters.")
        return
    
    st.header(f"📋 Search Results ({len(results)} found)")
    st.markdown(f"**Search Query:** *{query}*")
    
    for i, result in enumerate(results):
        with st.expander(f"📄 Result {i+1}: {result['filename']} ({result['period']}, {result['genre']})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                text = result['normalized_text']
                if len(text) > 800:
                    text = text[:800] + "..."
                
                highlighted_text = highlight_search_terms(text, search_terms)
                
                st.markdown(f"""
                <div class="result-card">
                    <p>{highlighted_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if result.get('original_text') and result['original_text'] != result['normalized_text']:
                    if st.button(f"📜 Show Original Text", key=f"orig_{i}"):
                        original = result['original_text']
                        if len(original) > 800:
                            original = original[:800] + "..."
                        st.text_area("Original Historical Text:", original, height=150, key=f"orig_text_{i}")
            
            with col2:
                st.markdown(f"""
                **📅 Period:** {result['period']}  
                **📚 Genre:** {result['genre']}  
                **🗓️ Year:** {result['year']}  
                **📄 Document:** {result['doc_id']}  
                **🔢 Tokens:** {result['token_count']}  
                **📁 File:** {result['filename']}
                """)

def render_statistics_dashboard():
    """Render the main statistics dashboard."""
    with st.spinner("Loading corpus statistics..."):
        stats = get_corpus_statistics()
    
    if not stats:
        st.warning("Could not load corpus statistics.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
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
            <h2>{stats['total_tokens']:,}</h2>
            <p>Total Tokens</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{stats['periods']}</h2>
            <p>Time Periods</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{stats['genres']}</h2>
            <p>Text Genres</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    📅 **Temporal Range:** {stats['earliest_year']} - {stats['latest_year']}  
    📊 **Average Tokens per Chunk:** {int(stats['avg_tokens'])}
    """)

def render_advanced_analytics():
    """Render the advanced analytics dashboard."""
    st.header("🔬 Advanced Linguistic Analytics")
    
    analysis_type = st.selectbox(
        "Choose Analysis Type:",
        [
            "📈 Word Frequency Timeline",
            "📚 Genre Vocabulary Analysis", 
            "🔗 Word Co-occurrence Analysis",
            "📜 Historical Spelling Evolution",
            "📊 Text Complexity Analysis",
            "📝 Word Frequency Analysis"
        ]
    )
    
    if analysis_type == "📈 Word Frequency Timeline":
        st.markdown("### Track how specific words evolved over time")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            word = st.text_input("Enter word to analyze:", value="gott")
        
        with col2:
            if st.button("🔍 Analyze Word"):
                if word:
                    with st.spinner(f"Analyzing '{word}'..."):
                        df = word_frequency_over_time(word)
                        
                        if df is not None and not df.empty:
                            fig = create_word_timeline_chart(df, word)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("### 📋 Detailed Results")
                            st.dataframe(df, use_container_width=True)
                            
                            total_freq = df['frequency'].sum()
                            peak_period = df.loc[df['frequency'].idxmax(), 'period']
                            st.markdown(f"""
                            <div class="insights-box">
                            <strong>🧠 Insights:</strong><br>
                            • Total occurrences of '{word}': <strong>{total_freq}</strong><br>
                            • Peak usage period: <strong>{peak_period}</strong><br>
                            • Found in genres: {', '.join(df['genres_found'].unique())}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning(f"No occurrences of '{word}' found in the corpus.")
    
    elif analysis_type == "📚 Genre Vocabulary Analysis":
        st.markdown("### Compare vocabulary richness and patterns across text genres")
        
        with st.spinner("Analyzing genre patterns..."):
            df = genre_vocabulary_analysis()
            
            if df is not None and not df.empty:
                fig = create_genre_comparison_chart(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 📊 Genre Statistics")
                cols = st.columns(min(len(df), 4))
                
                for i, (_, row) in enumerate(df.iterrows()):
                    with cols[i % len(cols)]:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>{row['genre']}</h4>
                            <p><strong>{row['total_tokens']:,}</strong> tokens</p>
                            <p><strong>{row['total_chunks']}</strong> chunks</p>
                            <p><strong>{row['avg_tokens_per_chunk']}</strong> avg/chunk</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("### 📋 Detailed Comparison")
                st.dataframe(df, use_container_width=True)
    
    elif analysis_type == "🔗 Word Co-occurrence Analysis":
        st.markdown("### Discover which words frequently appear together")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            word1 = st.text_input("First word:", value="gott")
        
        with col2:
            word2 = st.text_input("Second word:", value="himmel")
        
        with col3:
            if st.button("🔍 Analyze Co-occurrence"):
                if word1 and word2:
                    with st.spinner(f"Analyzing '{word1}' & '{word2}'..."):
                        df = co_occurrence_analysis(word1, word2)
                        
                        if df is not None and not df.empty:
                            fig = create_cooccurrence_heatmap(df, word1, word2)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("### 📋 Co-occurrence Details")
                            st.dataframe(df, use_container_width=True)
                            
                            total_cooccur = df['cooccurrences'].sum()
                            top_genre = df.loc[df['cooccurrences'].idxmax(), 'genre']
                            st.markdown(f"""
                            <div class="insights-box">
                            <strong>🧠 Insights:</strong><br>
                            • Total co-occurrences: <strong>{total_cooccur}</strong><br>
                            • Most common in: <strong>{top_genre}</strong> texts<br>
                            • Found across <strong>{df['period'].nunique()}</strong> time periods
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning(f"No co-occurrences found for '{word1}' and '{word2}'.")
    
    elif analysis_type == "📜 Historical Spelling Evolution":
        st.markdown("### Track archaic spelling patterns across time periods")
        
        with st.spinner("Analyzing spelling evolution..."):
            df = historical_spelling_analysis()
            
            if df is not None and not df.empty:
                fig = create_spelling_evolution_chart(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 📊 Archaic Spelling Percentage by Period")
                fig_percent = px.bar(
                    df, 
                    x='period', 
                    y='archaic_percentage',
                    title="Percentage of Texts with Archaic Spelling Patterns",
                    color='archaic_percentage',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_percent, use_container_width=True)
                
                st.markdown("### 📋 Detailed Spelling Analysis")
                st.dataframe(df, use_container_width=True)
    
    elif analysis_type == "📊 Text Complexity Analysis":
        st.markdown("### Analyze text complexity patterns across periods and genres")
        
        with st.spinner("Analyzing text complexity..."):
            df = text_complexity_analysis()
            
            if df is not None and not df.empty:
                fig = create_complexity_scatter(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                period_avg = df.groupby('period').agg({
                    'avg_word_length': 'mean',
                    'avg_word_count': 'mean',
                    'chunk_count': 'sum'
                }).round(2)
                
                st.markdown("### 📈 Complexity Trends by Period")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_length = px.line(
                        x=period_avg.index, 
                        y=period_avg['avg_word_length'],
                        title="Average Word Length Over Time",
                        labels={'x': 'Period', 'y': 'Average Word Length'}
                    )
                    st.plotly_chart(fig_length, use_container_width=True)
                
                with col2:
                    fig_count = px.line(
                        x=period_avg.index, 
                        y=period_avg['avg_word_count'],
                        title="Average Words per Chunk Over Time",
                        labels={'x': 'Period', 'y': 'Average Word Count'}
                    )
                    st.plotly_chart(fig_count, use_container_width=True)
                
                st.markdown("### 📋 Detailed Complexity Analysis")
                st.dataframe(df, use_container_width=True)
    
    elif analysis_type == "📝 Word Frequency Analysis":
        st.markdown("### Visualize most frequent words by time period")
        
        with st.spinner("Loading word frequency data..."):
            words_df = top_words_by_period()
            
            if words_df is not None and not words_df.empty:
                periods = words_df['period'].unique()
                selected_period = st.selectbox("Select time period:", periods)
                
                if selected_period:
                    period_data = words_df[words_df['period'] == selected_period].head(10)
                    
                    fig = px.bar(
                        period_data,
                        x='frequency',
                        y='word',
                        orientation='h',
                        title=f"Top 10 Words in {selected_period}",
                        labels={'frequency': 'Frequency', 'word': 'Word'},
                        color='frequency',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    render_word_frequency_grid(words_df, selected_period)
                    
                    period_words = words_df[words_df['period'] == selected_period]
                    st.markdown(f"### 📋 Complete Word List for {selected_period}")
                    st.dataframe(period_words[['word', 'frequency']], use_container_width=True)

def render_corpus_overview():
    """Render corpus overview with charts."""
    st.header("📊 Corpus Overview")
    
    with st.spinner("Loading corpus distribution..."):
        conn = get_database_connection()
        if not conn:
            st.error("Cannot connect to database")
            return
        
        try:
            period_query = """
            SELECT period, COUNT(*) as count
            FROM chunks
            GROUP BY period
            ORDER BY period
            """
            periods_df = pd.read_sql(period_query, conn)
            
            genre_query = """
            SELECT genre, COUNT(*) as count
            FROM chunks
            GROUP BY genre
            ORDER BY count DESC
            """
            genres_df = pd.read_sql(genre_query, conn)
            
            year_query = """
            SELECT year, COUNT(*) as count
            FROM chunks
            WHERE year IS NOT NULL
            GROUP BY year
            ORDER BY year
            """
            years_df = pd.read_sql(year_query, conn)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
        finally:
            conn.close()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Distribution by Period")
        fig_period = px.bar(
            periods_df, 
            x='period', 
            y='count',
            title="Chunks by Time Period",
            color='count',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig_period, use_container_width=True)
        
        st.write("**Period Details:**")
        for _, row in periods_df.iterrows():
            st.write(f"• {row['period']}: {row['count']:,} chunks")
    
    with col2:
        st.subheader("📚 Distribution by Genre")
        fig_genre = px.pie(
            genres_df,
            values='count',
            names='genre',
            title="Chunks by Genre"
        )
        st.plotly_chart(fig_genre, use_container_width=True)
        
        st.write("**Genre Details:**")
        for _, row in genres_df.iterrows():
            st.write(f"• {row['genre']}: {row['count']:,} chunks")
    
    if not years_df.empty:
        st.subheader("📈 Temporal Distribution by Year")
        fig_year = px.line(
            years_df,
            x='year',
            y='count',
            title="Text Chunks Over Time",
            markers=True
        )
        fig_year.update_layout(height=400)
        st.plotly_chart(fig_year, use_container_width=True)

# ==============================================
# NEW RAG FEATURES (ADDED WITHOUT BREAKING EXISTING)
# ==============================================

def render_semantic_search_tab():
    """NEW: Render semantic search capabilities."""
    st.header("🧠 AI-Powered Semantic Search")
    
    # Check RAG availability
    rag_manager = get_rag_manager()
    if rag_manager.initialize_rag():
        st.markdown("""
        <div class="rag-status-ready">
            ✅ <strong>AI Search System:</strong> Online and Ready for semantic queries
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="rag-card">
            <h3>🤖 Intelligent Semantic Search</h3>
            <p>Search by meaning and context, not just keywords. Ask questions about German language history!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example semantic queries
        st.markdown("**💡 Try these semantic search examples:**")
        semantic_examples = [
            "religious language in medieval texts",
            "how German spelling evolved over time", 
            "differences between legal and scientific writing",
            "archaic verb forms and grammar patterns",
            "biblical references in historical sermons"
        ]
        
        example_cols = st.columns(3)
        for i, example in enumerate(semantic_examples):
            if example_cols[i % 3].button(f"🔍 {example[:20]}...", key=f"sem_ex_{i}"):
                st.session_state['semantic_query'] = example
        
        # Semantic search input
        semantic_query = st.text_area(
            "Enter your semantic search query:",
            value=st.session_state.get('semantic_query', ''),
            placeholder="Describe concepts, ask questions, or search by meaning...",
            height=100,
            key="semantic_input"
        )
        
        # Clear session state
        if 'semantic_query' in st.session_state:
            del st.session_state['semantic_query']
        
        # Semantic search controls
        col1, col2 = st.columns(2)
        with col1:
            semantic_period = st.selectbox("Focus on period:", get_available_periods(), key="sem_period")
        with col2:
            semantic_k = st.slider("Number of results:", 3, 15, 8, key="sem_k")
        
        # Perform semantic search
        if st.button("🧠 Semantic Search", type="primary") and semantic_query.strip():
            with st.spinner("🤖 AI is analyzing semantic meaning..."):
                try:
                    results = rag_manager.semantic_search(
                        semantic_query, 
                        k=semantic_k, 
                        period_filter=semantic_period
                    )
                    
                    if results:
                        st.success(f"🎯 Found {len(results)} semantically relevant results!")
                        
                        for i, result in enumerate(results):
                            metadata = result.get('metadata', {})
                            similarity_score = 1 - result.get('distance', 0) if result.get('distance') else 0.95
                            
                            with st.expander(f"🤖 AI Match {i+1}: {metadata.get('filename', 'Unknown')} (Similarity: {similarity_score:.1%})"):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    text = result.get('text', '')
                                    if len(text) > 600:
                                        text = text[:600] + "..."
                                    
                                    st.markdown(f"""
                                    <div class="semantic-result">
                                        <p><strong>AI-Found Content:</strong></p>
                                        <p>{text}</p>
                                        <p><strong>🎯 Semantic Similarity: {similarity_score:.1%}</strong></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    **📅 Period:** {metadata.get('period', 'Unknown')}  
                                    **📚 Genre:** {metadata.get('genre', 'Unknown')}  
                                    **🗓️ Year:** {metadata.get('year', 'Unknown')}  
                                    **🔢 Words:** {metadata.get('word_count', 'Unknown')}
                                    """)
                    else:
                        st.warning("No semantic matches found. Try different search terms.")
                        
                except Exception as e:
                    st.error(f"Semantic search failed: {e}")
    
    else:
        st.markdown("""
        <div class="rag-status-error">
            ⚠️ <strong>AI Search System:</strong> Not Available
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **🔧 To enable AI-powered semantic search:**
        1. Run: `python rag.py --test --limit 500`
        2. Ensure vector database is created
        3. Restart this Streamlit app
        """)

# ==============================================
# MAIN APPLICATION (ENHANCED WITH RAG)
# ==============================================

def main():
    """Enhanced main application preserving ALL original features + adding RAG."""
    
    # Initialize session state
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ German Historical Corpus Analytics + AI</h1>
        <p>Complete SQL Analytics • Advanced Visualizations • AI-Powered Semantic Search</p>
        <p><small>All Original Features Preserved + New AI Capabilities</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check database connection
    conn = get_database_connection()
    if not conn:
        st.error("❌ Cannot connect to database. Please check your PostgreSQL connection.")
        st.stop()
    conn.close()
    
    # Statistics Dashboard (PRESERVED)
    render_statistics_dashboard()
    
    st.markdown("---")
    
    # Enhanced Navigation (PRESERVED + NEW)
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Traditional Search", 
        "🧠 AI Semantic Search", 
        "🔬 Advanced Analytics", 
        "📊 Corpus Overview"
    ])
    
    with tab1:
        # PRESERVED: Original search functionality
        st.header("🔍 Search Historical German Texts")
        
        # Example queries (PRESERVED)
        st.markdown("**💡 Try these example queries:**")
        examples = ["Gott", "König", "Krieg", "Recht", "Liebe", "Himmel", "thun", "vmb"]
        
        example_buttons = st.columns(len(examples))
        for i, example in enumerate(examples):
            if example_buttons[i].button(f"'{example}'", key=f"ex_{i}"):
                st.session_state['example_query'] = example
        
        # Search input (PRESERVED)
        query = st.text_area(
            "Enter your search query:",
            value=st.session_state.get('example_query', ''),
            placeholder="Enter German words or phrases to search for...",
            height=100,
            key="search_input"
        )
        
        # Clear example query after use
        if 'example_query' in st.session_state:
            del st.session_state['example_query']
        
        # Filters in sidebar (PRESERVED)
        st.sidebar.header("🎛️ Search Filters")
        
        periods = get_available_periods()
        genres = get_available_genres()
        
        period_filter = st.sidebar.selectbox("Time Period:", periods)
        genre_filter = st.sidebar.selectbox("Genre:", genres)
        limit = st.sidebar.slider("Max Results:", 5, 50, 10)
        
        # Search button (PRESERVED)
        if st.button("🔍 Search", type="primary") and query.strip():
            with st.spinner("Searching historical corpus..."):
                results, search_terms = search_database(query, period_filter, genre_filter, limit)
                render_search_results(results, search_terms, query)
                
                # Add to search history (PRESERVED)
                if results:
                    st.session_state.search_history.append({
                        'query': query,
                        'results': len(results),
                        'period': period_filter,
                        'genre': genre_filter
                    })
        
        # Search history in sidebar (PRESERVED)
        if st.session_state.search_history:
            st.sidebar.markdown("---")
            st.sidebar.header("📝 Recent Searches")
            for i, search in enumerate(st.session_state.search_history[-5:]):
                st.sidebar.write(f"• {search['query'][:20]}... ({search['results']} results)")
    
    with tab2:
        # NEW: AI Semantic Search (without breaking existing features)
        render_semantic_search_tab()
    
    with tab3:
        # PRESERVED: All original advanced analytics
        render_advanced_analytics()
    
    with tab4:
        # PRESERVED: Corpus overview
        render_corpus_overview()
    
    # Footer (ENHANCED)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏛️ German Historical Corpus Analytics + AI | Complete Feature Preservation + AI Enhancement</p>
        <p><small>Traditional SQL Analytics • Advanced Visualizations • AI Semantic Search • Historical Analysis</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()