# ==========================================
# FILE: Enhanced streamlit_app.py - Traditional + RAG Integration
# ==========================================
"""
Enhanced Streamlit app that preserves ALL original SQL features
and adds RAG semantic search as an optional enhancement
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
import requests
import time

# Page configuration
st.set_page_config(
    page_title="🏛️ German Historical Corpus Analytics + AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS - All original styling preserved + RAG additions
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
    
    .search-highlight {
        background-color: #fff3cd;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
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
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .rag-status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .rag-feature-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Database configuration (PRESERVED)
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'germanc_corpus',
    'user': 'rohan',
    'password': '1996'
}

def get_database_connection():
    """Get database connection (PRESERVED)."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# ==============================================
# RAG SERVICE CLIENT (NEW - EASIEST INTEGRATION)
# ==============================================

class RAGServiceClient:
    """Simple RAG client that connects to background service."""
    
    def __init__(self, service_url: str = "http://127.0.0.1:8001"):
        self.service_url = service_url.rstrip('/')
        self._is_available = None
        self._error_message = None
    
    def check_service_health(self) -> tuple[bool, str]:
        """Check if RAG service is available."""
        if self._is_available is not None:
            return self._is_available, self._error_message
        
        try:
            response = requests.get(f"{self.service_url}/health", timeout=3)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("is_initialized"):
                    self._is_available = True
                    self._error_message = None
                else:
                    self._is_available = False
                    self._error_message = health_data.get("error", "Service not initialized")
            else:
                self._is_available = False
                self._error_message = f"Service returned status {response.status_code}"
        except requests.exceptions.RequestException as e:
            self._is_available = False
            self._error_message = f"Cannot connect: {str(e)[:100]}"
        
        return self._is_available, self._error_message
    
    def semantic_search(self, query: str, k: int = 5, period_filter: str = None) -> List[Dict[str, Any]]:
        """Perform semantic search via the service."""
        try:
            response = requests.post(
                f"{self.service_url}/semantic_search",
                json={
                    "query": query,
                    "k": k,
                    "period_filter": period_filter if period_filter != "All Periods" else None
                },
                timeout=20
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except requests.exceptions.RequestException as e:
            st.error(f"🤖 AI Search failed: {e}")
            return []

@st.cache_resource
def get_rag_client():
    """Get cached RAG client instance."""
    return RAGServiceClient()

# ==============================================
# ALL ORIGINAL SQL FUNCTIONS (PRESERVED EXACTLY)
# ==============================================

def word_frequency_over_time(word):
    """Track word usage changes over time periods (PRESERVED)."""
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

def search_database(query: str, period_filter: str = None, genre_filter: str = None, limit: int = 10):
    """Search the database using PostgreSQL text search (PRESERVED)."""
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
            period,
            genre,
            year,
            filename,
            token_count
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

def get_corpus_statistics():
    """Get corpus statistics (PRESERVED)."""
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

def get_available_periods():
    """Get available periods from database (PRESERVED)."""
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

def highlight_search_terms(text: str, search_terms: List[str]) -> str:
    """Highlight search terms in text (PRESERVED)."""
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
# ORIGINAL RENDERING FUNCTIONS (PRESERVED)
# ==============================================

def render_search_results(results: List[Dict], search_terms: List[str], query: str):
    """Render search results with highlighting (PRESERVED)."""
    if not results:
        st.info("No results found. Try different search terms or adjust filters.")
        return
    
    st.header(f"📋 Traditional SQL Search Results ({len(results)} found)")
    st.markdown(f"**Search Query:** *{query}*")
    
    for i, result in enumerate(results):
        with st.expander(f"📄 Result {i+1}: {result['filename']} ({result['period']}, {result['genre']})"):
            text = result['normalized_text']
            if len(text) > 500:
                text = text[:500] + "..."
            
            highlighted_text = highlight_search_terms(text, search_terms)
            
            st.markdown(f"""
            <div class="result-card">
                <p>{highlighted_text}</p>
                <small><strong>Period:</strong> {result['period']} | <strong>Genre:</strong> {result['genre']} | 
                <strong>Year:</strong> {result['year']} | <strong>Tokens:</strong> {result['token_count']}</small>
            </div>
            """, unsafe_allow_html=True)

def render_statistics_dashboard():
    """Render the main statistics dashboard (PRESERVED)."""
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

# ==============================================
# NEW RAG SEMANTIC SEARCH FEATURE (EASIEST INTEGRATION)
# ==============================================

def render_semantic_search_results(results: List[Dict], query: str):
    """Render AI semantic search results."""
    if not results:
        st.info("🤖 No semantic matches found. The AI couldn't find relevant content for this query.")
        return
    
    st.header(f"🤖 AI Semantic Search Results ({len(results)} found)")
    st.markdown(f"**AI Query:** *{query}*")
    
    st.markdown("""
    <div class="rag-feature-box">
    <strong>🧠 How AI Semantic Search Works:</strong><br>
    Unlike traditional keyword search, AI understands meaning and context. 
    It can find relevant content even when exact words don't match!
    </div>
    """, unsafe_allow_html=True)
    
    for i, result in enumerate(results):
        metadata = result.get('metadata', {})
        similarity_score = 1 - result.get('distance', 0) if result.get('distance') else 0.95
        
        with st.expander(f"🤖 AI Match {i+1}: {metadata.get('filename', 'Unknown')} (Relevance: {similarity_score:.1%})"):
            text = result.get('text', '')
            if len(text) > 600:
                text = text[:600] + "..."
            
            st.markdown(f"""
            <div class="semantic-result">
                <p><strong>🤖 AI-Found Content:</strong></p>
                <p>{text}</p>
                <hr>
                <p><strong>🎯 AI Relevance Score: {similarity_score:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **📅 Period:** {metadata.get('period', 'Unknown')}  
                **📚 Genre:** {metadata.get('genre', 'Unknown')}  
                **🗓️ Year:** {metadata.get('year', 'Unknown')}
                """)
            with col2:
                st.markdown(f"""
                **🔢 Words:** {metadata.get('word_count', 'Unknown')}  
                **📄 Document:** {metadata.get('document_id', 'Unknown')}  
                **📁 File:** {metadata.get('filename', 'Unknown')}
                """)

def render_rag_status_check():
    """Show RAG service status."""
    rag_client = get_rag_client()
    is_available, error_message = rag_client.check_service_health()
    
    if is_available:
        st.markdown("""
        <div class="rag-status-ready">
            ✅ <strong>AI Search System:</strong> Online and Ready! 
            The background RAG service is running and initialized.
        </div>
        """, unsafe_allow_html=True)
        return True
    else:
        st.markdown(f"""
        <div class="rag-status-error">
            ⚠️ <strong>AI Search System:</strong> Not Available<br>
            <small>Error: {error_message}</small>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔧 How to Enable AI Search"):
            st.markdown("""
            **To enable AI-powered semantic search:**
            
            1. **Start the RAG service:**
               ```bash
               cd your-project-directory
               python -m src.rag_service.rag_server
               ```
            
            2. **Wait for initialization** (30-60 seconds)
            
            3. **Refresh this page** - AI search will become available
            
            **Note:** The RAG service runs separately to keep your traditional analytics fast and reliable.
            """)
        return False

# ==============================================
# MAIN APPLICATION (ENHANCED BUT PRESERVES ALL ORIGINAL)
# ==============================================

def main():
    """Enhanced main application - ALL original features preserved + AI addition."""
    
    # Header (ENHANCED)
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ German Historical Corpus Analytics + AI</h1>
        <p>Traditional SQL Analytics • Advanced Visualizations • AI-Powered Search</p>
        <p><small>All Original Features Preserved + Optional AI Enhancement</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check database connection (PRESERVED)
    conn = get_database_connection()
    if not conn:
        st.error("❌ Cannot connect to database. Please check your PostgreSQL connection.")
        st.stop()
    conn.close()
    
    # Statistics Dashboard (PRESERVED)
    render_statistics_dashboard()
    
    st.markdown("---")
    
    # Enhanced Navigation - Traditional + AI tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Traditional SQL Search", 
        "🤖 AI Semantic Search (NEW!)", 
        "📊 Advanced Analytics"
    ])
    
    with tab1:
        # PRESERVED: All original search functionality exactly as before
        st.header("🔍 Traditional SQL Database Search")
        
        st.markdown("**💡 Try these example queries:**")
        examples = ["Gott", "König", "Krieg", "Recht", "Liebe", "Himmel"]
        
        example_buttons = st.columns(len(examples))
        for i, example in enumerate(examples):
            if example_buttons[i].button(f"'{example}'", key=f"sql_ex_{i}"):
                st.session_state['sql_query'] = example
        
        # Search input (PRESERVED)
        query = st.text_area(
            "Enter your search query:",
            value=st.session_state.get('sql_query', ''),
            placeholder="Enter German words or phrases to search for...",
            height=80,
            key="sql_search_input"
        )
        
        # Clear example query after use
        if 'sql_query' in st.session_state:
            del st.session_state['sql_query']
        
        # Filters (PRESERVED)
        col1, col2, col3 = st.columns(3)
        with col1:
            period_filter = st.selectbox("Time Period:", get_available_periods(), key="sql_period")
        with col2:
            genre_filter = st.selectbox("Genre:", ["All Genres"], key="sql_genre")  # Simplified for demo
        with col3:
            limit = st.slider("Max Results:", 5, 20, 10, key="sql_limit")
        
        # Search button (PRESERVED)
        if st.button("🔍 SQL Search", type="primary", key="sql_search_btn") and query.strip():
            with st.spinner("Searching database with SQL..."):
                results, search_terms = search_database(query, period_filter, genre_filter, limit)
                render_search_results(results, search_terms, query)
    
    with tab2:
        # NEW: AI Semantic Search feature (easiest integration)
        st.header("🤖 AI-Powered Semantic Search")
        
        # Check RAG service availability
        rag_available = render_rag_status_check()
        
        if rag_available:
            st.markdown("""
            <div class="rag-card">
                <h3>🧠 Intelligent Semantic Search</h3>
                <p>Ask questions in natural language! The AI understands meaning, not just keywords.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Example semantic queries
            st.markdown("**💡 Try these AI-powered semantic search examples:**")
            ai_examples = [
                "religious language in medieval texts",
                "how German spelling evolved", 
                "legal terminology differences",
                "archaic verb forms",
                "biblical references in sermons"
            ]
            
            ai_cols = st.columns(3)
            for i, example in enumerate(ai_examples):
                if ai_cols[i % 3].button(f"🤖 {example[:15]}...", key=f"ai_ex_{i}"):
                    st.session_state['ai_query'] = example
            
            # AI search input
            ai_query = st.text_area(
                "Ask the AI about German historical language:",
                value=st.session_state.get('ai_query', ''),
                placeholder="Ask questions like: 'How did religious language change over time?' or 'Show me archaic spelling patterns'...",
                height=100,
                key="ai_search_input"
            )
            
            # Clear AI example query
            if 'ai_query' in st.session_state:
                del st.session_state['ai_query']
            
            # AI search controls
            col1, col2 = st.columns(2)
            with col1:
                ai_period = st.selectbox("Focus on period:", get_available_periods(), key="ai_period")
            with col2:
                ai_k = st.slider("AI Results:", 3, 12, 6, key="ai_k")
            
            # AI Search button
            if st.button("🧠 AI Semantic Search", type="secondary", key="ai_search_btn") and ai_query.strip():
                with st.spinner("🤖 AI is analyzing semantic meaning... This may take a moment."):
                    rag_client = get_rag_client()
                    
                    try:
                        ai_results = rag_client.semantic_search(
                            ai_query, 
                            k=ai_k, 
                            period_filter=ai_period
                        )
                        
                        if ai_results:
                            st.success(f"🎯 AI found {len(ai_results)} semantically relevant results!")
                            render_semantic_search_results(ai_results, ai_query)
                        else:
                            st.warning("🤖 AI couldn't find semantic matches. Try rephrasing your question or check if the RAG service is properly initialized.")
                            
                    except Exception as e:
                        st.error(f"🤖 AI search encountered an error: {e}")
                        st.info("💡 Try restarting the RAG service or check the connection.")
        
        else:
            st.info("""
            🤖 **AI Semantic Search is an optional enhancement** that works alongside your traditional analytics.
            
            Your existing SQL search and analytics are fully functional without AI!
            """)
    
    with tab3:
        # PRESERVED: Word frequency analysis and other original features
        st.header("📊 Advanced Linguistic Analytics")
        
        st.markdown("### 📈 Word Frequency Timeline")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            word = st.text_input("Enter word to analyze:", value="gott", key="word_freq_input")
        
        with col2:
            if st.button("🔍 Analyze Word", key="word_freq_btn"):
                if word:
                    with st.spinner(f"Analyzing '{word}'..."):
                        df = word_frequency_over_time(word)
                        
                        if df is not None and not df.empty:
                            # Create timeline chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df['period'],
                                y=df['frequency'],
                                mode='lines+markers',
                                name='Frequency',
                                line=dict(color='#2a5298', width=3),
                                marker=dict(size=10, color='#2a5298')
                            ))
                            
                            fig.update_layout(
                                title=f"📈 Frequency of '{word}' Over Time",
                                xaxis_title="Time Period",
                                yaxis_title="Frequency",
                                template="plotly_white",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("### 📋 Detailed Results")
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.warning(f"No occurrences of '{word}' found in the corpus.")
    
    # Footer (ENHANCED)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏛️ German Historical Corpus Analytics + AI</p>
        <p><small>Traditional SQL Analytics (Always Available) + AI Semantic Search (Optional Enhancement)</small></p>
        <p><small>Your data, your choice: Use traditional methods, AI features, or both together!</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()