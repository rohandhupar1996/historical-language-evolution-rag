#!/bin/bash
# ==========================================
# FILE: start_full_system.sh
# ==========================================
# Complete system startup script

echo "🏛️ Starting Complete GerManC System..."
echo "======================================"

# Function to check if port is in use
check_port() {
    if lsof -i :$1 > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found!"
    exit 1
fi

# Check PostgreSQL
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "❌ PostgreSQL not running!"
    echo "📝 Start with: brew services start postgresql"
    exit 1
fi

# Check vector database
if [ ! -d "./german_corpus_vectordb" ]; then
    echo "❌ Vector database not found!"
    echo "📝 Run: python rag.py --test --limit 1000"
    exit 1
fi

echo "✅ Prerequisites OK!"

# Kill existing services if running
if check_port 8001; then
    echo "🔄 Stopping existing RAG service on port 8001..."
    kill $(lsof -t -i:8001) 2>/dev/null || true
    sleep 2
fi

if check_port 8501; then
    echo "🔄 Stopping existing Streamlit on port 8501..."
    kill $(lsof -t -i:8501) 2>/dev/null || true
    sleep 2
fi

echo ""
echo "🚀 Starting RAG service in background..."
python -m src.rag_service.rag_server &
RAG_PID=$!

# Wait for RAG service to be ready
echo "⏳ Waiting for RAG service to initialize..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8001/health > /dev/null 2>&1; then
        echo "✅ RAG service ready!"
        break
    fi
    echo "   Attempt $i/30..."
    sleep 2
done

# Check if RAG service is actually ready
if ! curl -s http://127.0.0.1:8001/health | grep -q '"is_initialized":true'; then
    echo "❌ RAG service failed to initialize properly!"
    echo "🔍 Check the logs above for errors"
    kill $RAG_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "🌐 Starting Streamlit web application..."
streamlit run streamlit_app.py &
STREAMLIT_PID=$!

echo ""
echo "🎉 Complete system started successfully!"
echo "======================================"
echo "🤖 RAG Service: http://127.0.0.1:8001"
echo "🌐 Web App: http://localhost:8501"
echo ""
echo "💡 To stop the system:"
echo "   Press Ctrl+C or run: ./stop_system.sh"
echo "======================================"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down system..."
    kill $RAG_PID 2>/dev/null || true
    kill $STREAMLIT_PID 2>/dev/null || true
    echo "✅ System stopped"
}

# Trap Ctrl+C
trap cleanup EXIT

# Wait for processes
wait