#!/bin/bash
# ==========================================
# FILE: start_streamlit.sh  
# ==========================================
# Startup script for Streamlit app

echo "🌐 Starting GerManC Streamlit Web Application..."
echo "=============================================="

# Check if RAG service is running
if ! curl -s http://127.0.0.1:8001/health > /dev/null 2>&1; then
    echo "⚠️  RAG service not detected on port 8001"
    echo "💡 For AI semantic search, start RAG service first:"
    echo "   ./start_rag_service.sh"
    echo ""
    echo "🔄 Starting Streamlit without AI features..."
else
    echo "✅ RAG service detected - AI features will be available!"
fi

echo ""
echo "🌐 Starting Streamlit on port 8501..."
echo "🔗 Web app will be available at: http://localhost:8501"
echo "=============================================="

# Start Streamlit
streamlit run streamlit_app.py