# ==========================================
# FILE: Fixed streamlit_app.py - No Warnings + Input Validation
# ==========================================
"""
Complete Streamlit app with fixes:
1. SQLAlchemy connections (no more pandas warnings)
2. Input validation for AI questions
3. Real AI status checking
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
import requests
from sqlalchemy import create_engine  # FIX for pandas warnings
import time

# Page configuration
st.set_page_config(
    page_title="🏛️ German Historical Corpus Analytics + AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS (preserved)
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
    
    .ai-answer-box {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #00b894;
    }
    
    .ai-status-online {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .ai-status-offline {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .validation-error {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
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

# FIXED: Create SQLAlchemy engine to eliminate pandas warnings
@st.cache_resource
def get_sqlalchemy_engine():
    """Create SQLAlchemy engine for pandas (eliminates warnings)."""
    connection_string = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(connection_string)

def get_database_connection():
    """Get psycopg2 connection for non-pandas operations."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# ==============================================
# AI RAG CLIENT WITH REAL STATUS CHECKING
# ==============================================

class SimpleRAGClient:
    """Simple RAG client with real status checking."""
    
    def __init__(self, url="http://127.0.0.1:8001"):
        self.url = url
        self._last_check = 0
        self._cached_status = False
        self._cache_duration = 30  # Cache status for 30 seconds
    
    def is_available(self):
        """Check if AI service is REALLY available (with caching)."""
        current_time = time.time()
        
        # Use cached result if recent
        if current_time - self._last_check < self._cache_duration:
            return self._cached_status
        
        try:
            response = requests.get(f"{self.url}/health", timeout=3)
            if response.status_code == 200:
                health_data = response.json()
                self._cached_status = health_data.get("is_initialized", False)
            else:
                self._cached_status = False
        except requests.exceptions.RequestException:
            self._cached_status = False
        
        self._last_check = current_time
        return self._cached_status
    
    def ask_question(self, question, period_filter=None):
        """Ask AI a question with real API call."""
        try:
            data = {"question": question}
            if period_filter and period_filter != "All Periods":
                data["period_filter"] = period_filter
            
            response = requests.post(f"{self.url}/ask_question", json=data, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"AI API Error: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"AI Connection Error: {e}")
            return None

# ==============================================
# INPUT VALIDATION FUNCTIONS
# ==============================================

def validate_search_input(query: str) -> tuple[bool, str]:
    """Validate search input and return (is_valid, error_message)."""
    if not query or not query.strip():
        return False, "⚠️ Please enter a search query. Empty searches are not allowed."
    
    if len(query.strip()) < 2:
        return False, "⚠️ Search query must be at least 2 characters long."
    
    if len(query.strip()) > 500:
        return False, "⚠️ Search query is too long. Please limit to 500 characters."
    
    # Check for potentially harmful input
    harmful_patterns = [';', '--', 'DROP', 'DELETE', 'INSERT', 'UPDATE']
    query_upper = query.upper()
    if any(pattern in query_upper for pattern in harmful_patterns):
        return False, "⚠️ Invalid characters detected. Please use only letters, numbers, and basic punctuation."
    
    return True, ""

def validate_ai_question(question: str) -> tuple[bool, str]:
    """Validate AI question input."""
    if not question or not question.strip():
        return False, "🤖 Please ask a question! AI needs something to analyze."
    
    if len(question.strip()) < 5:
        return False, "🤖 Please ask a more detailed question (at least 5 characters)."
    
    if len(question.strip()) > 1000:
        return False, "🤖 Question is too long. Please limit to 1000 characters."
    
    # Basic question validation
    if not any(char in question.lower() for char in ['?', 'how', 'what', 'when', 'where', 'why', 'describe', 'explain', 'show', 'tell']):
        return False, "🤖 Please phrase as a question or request (use words like 'how', 'what', 'describe', etc.)"
    
    return True, ""

# ==============================================
# FIXED SQL FUNCTIONS (NO MORE PANDAS WARNINGS)
# ==============================================

def word_frequency_over_time(word):
    """Track word usage changes over time periods (FIXED)."""
    engine = get_sqlalchemy_engine()
    
    try:
        query = """
        SELECT 
            period, 
            COUNT(*) as frequency,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage,
            STRING_AGG(DISTINCT genre, ', ') as genres_found
        FROM chunks 
        WHERE normalized_text ILIKE %(word)s
        GROUP BY period 
        ORDER BY period
        """
        df = pd.read_sql(query, engine, params={'word': f'%{word}%'})
        return df
    except Exception as e:
        st.error(f"Error in word frequency analysis: {e}")
        return None

def get_corpus_statistics():
    """Get corpus statistics (FIXED)."""
    engine = get_sqlalchemy_engine()
    
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
        df = pd.read_sql(query, engine)
        return df.iloc[0].to_dict()
    except Exception as e:
        st.error(f"Error fetching statistics: {e}")
        return None

def search_database(query: str, period_filter: str = None, genre_filter: str = None, limit: int = 10):
    """Search the database (FIXED)."""
    engine = get_sqlalchemy_engine()
    
    try:
        search_terms = query.split()
        where_conditions = []
        params = {}
        
        for i, term in enumerate(search_terms):
            where_conditions.append(f"normalized_text ILIKE %(term_{i})s")
            params[f'term_{i}'] = f"%{term}%"
        
        search_condition = " AND ".join(where_conditions)
        
        if period_filter and period_filter != "All Periods":
            search_condition += " AND period = %(period_filter)s"
            params['period_filter'] = period_filter
        
        if genre_filter and genre_filter != "All Genres":
            search_condition += " AND genre = %(genre_filter)s"
            params['genre_filter'] = genre_filter
        
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
        LIMIT %(limit)s
        """
        
        params['limit'] = limit
        df = pd.read_sql(search_query, engine, params=params)
        
        return df.to_dict('records'), search_terms
    except Exception as e:
        st.error(f"Search error: {e}")
        return [], []

def get_available_periods():
    """Get available periods (FIXED)."""
    engine = get_sqlalchemy_engine()
    
    try:
        query = "SELECT DISTINCT period FROM chunks ORDER BY period"
        df = pd.read_sql(query, engine)
        periods = ["All Periods"] + df['period'].tolist()
        return periods
    except Exception:
        return ["All Periods"]

def get_available_genres():
    """Get available genres (FIXED)."""
    engine = get_sqlalchemy_engine()
    
    try:
        query = "SELECT DISTINCT genre FROM chunks ORDER BY genre"
        df = pd.read_sql(query, engine)
        genres = ["All Genres"] + df['genre'].tolist()
        return genres
    except Exception:
        return ["All Genres"]

def render_corpus_overview():
    """Render corpus overview (FIXED)."""
    st.header("📊 Corpus Overview")
    
    with st.spinner("Loading corpus distribution..."):
        engine = get_sqlalchemy_engine()
        
        try:
            period_query = """
            SELECT period, COUNT(*) as count
            FROM chunks
            GROUP BY period
            ORDER BY period
            """
            periods_df = pd.read_sql(period_query, engine)
            
            genre_query = """
            SELECT genre, COUNT(*) as count
            FROM chunks
            GROUP BY genre
            ORDER BY count DESC
            """
            genres_df = pd.read_sql(genre_query, engine)
            
            year_query = """
            SELECT year, COUNT(*) as count
            FROM chunks
            WHERE year IS NOT NULL
            GROUP BY year
            ORDER BY year
            """
            years_df = pd.read_sql(year_query, engine)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
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
    
    with col2:
        st.subheader("📚 Distribution by Genre")
        fig_genre = px.pie(
            genres_df,
            values='count',
            names='genre',
            title="Chunks by Genre"
        )
        st.plotly_chart(fig_genre, use_container_width=True)

# ==============================================
# OTHER PRESERVED FUNCTIONS (keeping original logic)
# ==============================================

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

def create_word_timeline_chart(df, word):
    """Create word timeline chart."""
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

def render_advanced_analytics():
    """Render advanced analytics with input validation."""
    st.header("🔬 Advanced Linguistic Analytics")
    
    st.markdown("### 📈 Word Frequency Timeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        word = st.text_input("Enter word to analyze:", value="gott")
    
    with col2:
        if st.button("🔍 Analyze Word"):
            # Validate input
            is_valid, error_msg = validate_search_input(word)
            if not is_valid:
                st.markdown(f"""
                <div class="validation-error">
                    {error_msg}
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner(f"Analyzing '{word}'..."):
                    df = word_frequency_over_time(word)
                    
                    if df is not None and not df.empty:
                        fig = create_word_timeline_chart(df, word)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning(f"No occurrences of '{word}' found in the corpus.")

# ==============================================
# MAIN APPLICATION
# ==============================================

def main():
    """Enhanced main application with fixes."""
    
    # Initialize session state
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ German Historical Corpus Analytics + AI</h1>
        <p>Complete SQL Analytics • Advanced Visualizations • AI-Powered Analysis</p>
        <p><small>All Original Features Preserved + New AI Enhancement</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check database connection
    conn = get_database_connection()
    if not conn:
        st.error("❌ Cannot connect to database. Please check your PostgreSQL connection.")
        st.stop()
    conn.close()
    
    # Statistics Dashboard
    render_statistics_dashboard()
    
    st.markdown("---")
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Traditional Search", 
        "🔬 Advanced Analytics", 
        "📊 Corpus Overview",
        "🤖 AI Analysis"
    ])
    
    with tab1:
        st.header("🔍 Search Historical German Texts")
        
        # Example queries
        st.markdown("**💡 Try these example queries:**")
        examples = ["Gott", "König", "Krieg", "Recht", "Liebe", "Himmel"]
        
        example_buttons = st.columns(len(examples))
        for i, example in enumerate(examples):
            if example_buttons[i].button(f"'{example}'", key=f"ex_{i}"):
                st.session_state['example_query'] = example
        
        # Search input with validation
        query = st.text_area(
            "Enter your search query:",
            value=st.session_state.get('example_query', ''),
            placeholder="Enter German words or phrases to search for...",
            height=100,
            key="search_input"
        )
        
        if 'example_query' in st.session_state:
            del st.session_state['example_query']
        
        # Filters
        st.sidebar.header("🎛️ Search Filters")
        periods = get_available_periods()
        genres = get_available_genres()
        
        period_filter = st.sidebar.selectbox("Time Period:", periods)
        genre_filter = st.sidebar.selectbox("Genre:", genres)
        limit = st.sidebar.slider("Max Results:", 5, 50, 10)
        
        # Search with validation
        if st.button("🔍 Search", type="primary"):
            is_valid, error_msg = validate_search_input(query)
            if not is_valid:
                st.markdown(f"""
                <div class="validation-error">
                    {error_msg}
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("Searching historical corpus..."):
                    results, search_terms = search_database(query, period_filter, genre_filter, limit)
                    render_search_results(results, search_terms, query)
                    
                    if results:
                        st.session_state.search_history.append({
                            'query': query,
                            'results': len(results),
                            'period': period_filter,
                            'genre': genre_filter
                        })
        
        # Search history
        if st.session_state.search_history:
            st.sidebar.markdown("---")
            st.sidebar.header("📝 Recent Searches")
            for i, search in enumerate(st.session_state.search_history[-5:]):
                st.sidebar.write(f"• {search['query'][:20]}... ({search['results']} results)")
    
    with tab2:
        render_advanced_analytics()
    
    with tab3:
        render_corpus_overview()
    
    with tab4:
        st.header("🤖 AI Historical Language Analysis")
        
        # Real AI status check
        rag_client = SimpleRAGClient()
        ai_available = rag_client.is_available()
        
        if ai_available:
            st.markdown("""
            <div class="ai-status-online">
                ✅ <strong>AI System Online:</strong> Ask questions about German historical language evolution! 
                (Status verified: RAG server is running and initialized)
            </div>
            """, unsafe_allow_html=True)
            
            # Example questions
            st.markdown("**💡 Try these AI-powered questions:**")
            ai_examples = [
                "How did German spelling change over time?",
                "What are characteristics of medieval German texts?", 
                "Describe religious language in historical documents",
                "How did legal terminology evolve?",
                "What archaic words were commonly used?"
            ]
            
            example_cols = st.columns(2)
            for i, example in enumerate(ai_examples):
                if example_cols[i % 2].button(f"❓ {example[:30]}...", key=f"ai_q_{i}"):
                    st.session_state['ai_question'] = example
            
            # AI Question input with validation
            question = st.text_area(
                "Ask AI about German historical language:",
                value=st.session_state.get('ai_question', ''),
                placeholder="Ask questions like: 'How did religious language change?' or 'What are common archaic spelling patterns?'",
                height=100,
                key="ai_question_input"
            )
            
            if 'ai_question' in st.session_state:
                del st.session_state['ai_question']
            
            # AI Controls
            col1, col2 = st.columns(2)
            with col1:
                ai_period = st.selectbox("Focus on period:", get_available_periods(), key="ai_period")
            with col2:
                if st.button("🤖 Ask AI", type="primary", key="ai_ask_btn"):
                    # Validate AI question
                    is_valid, error_msg = validate_ai_question(question)
                    if not is_valid:
                        st.markdown(f"""
                        <div class="validation-error">
                            {error_msg}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        with st.spinner("🤖 AI is analyzing historical texts..."):
                            result = rag_client.ask_question(question, ai_period)
                            
                            if result:
                                # Display AI Answer
                                st.markdown("### 🤖 AI Analysis")
                                st.markdown(f"""
                                <div class="ai-answer-box">
                                    <p><strong>Question:</strong> {question}</p>
                                    <hr>
                                    <p><strong>AI Answer:</strong> {result['answer']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display Sources
                                if result.get('source_documents'):
                                    st.markdown("### 📚 Historical Sources")
                                    st.info(f"AI analyzed {len(result['source_documents'])} historical sources")
                                    
                                    for i, doc in enumerate(result['source_documents'][:3]):
                                        with st.expander(f"📜 Source {i+1}"):
                                            source_text = doc['content']
                                            if len(source_text) > 500:
                                                source_text = source_text[:500] + "..."
                                            
                                            st.text_area("Historical Text:", source_text, height=120, key=f"ai_source_{i}")
                                            
                                            if doc.get('metadata'):
                                                metadata = doc['metadata']
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.write(f"**Period:** {metadata.get('period', 'Unknown')}")
                                                with col2:
                                                    st.write(f"**Genre:** {metadata.get('genre', 'Unknown')}")
                                                with col3:
                                                    st.write(f"**Year:** {metadata.get('year', 'Unknown')}")
                            else:
                                st.error("🤖 AI service unavailable. Please check if RAG server is running.")
        
        else:
            st.markdown("""
            <div class="ai-status-offline">
                ⚠️ <strong>AI Analysis System:</strong> Currently Offline 
                (Real status check: Cannot connect to RAG server at http://127.0.0.1:8001)
            </div>
            """, unsafe_allow_html=True)
            
            st.info("**🤖 AI Analysis is optional.** Your traditional SQL analytics work perfectly without AI!")
            
            with st.expander("🔧 How to Enable AI"):
                st.code("python -m src.rag_service.rag_server", language="bash")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏛️ German Historical Corpus Analytics + AI</p>
        <p><small>Traditional SQL Analytics (Always Available) + AI Analysis (Optional Enhancement)</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()