#!/bin/bash
# ==========================================
# FILE: stop_system.sh
# ==========================================
# Stop all services

echo "🛑 Stopping GerManC System..."

# Stop RAG service
if lsof -i :8001 > /dev/null 2>&1; then
    echo "🔄 Stopping RAG service on port 8001..."
    kill $(lsof -t -i:8001) 2>/dev/null || true
fi

# Stop Streamlit
if lsof -i :8501 > /dev/null 2>&1; then
    echo "🔄 Stopping Streamlit on port 8501..."
    kill $(lsof -t -i:8501) 2>/dev/null || true
fi

echo "✅ System stopped"

---

@echo off
REM ==========================================
REM FILE: start_rag_service.bat (Windows)
REM ==========================================

echo 🚀 Starting GerManC RAG Background Service...
echo ==============================================

REM Check if vector database exists
if not exist "german_corpus_vectordb" (
    echo ❌ Vector database not found!
    echo 📝 Please run first: python rag.py --test --limit 1000
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed!
echo.
echo 🔄 Starting RAG service on port 8001...
echo 💡 Keep this window open - the service needs to run continuously
echo 🌐 Service will be available at: http://127.0.0.1:8001
echo.
echo Press Ctrl+C to stop the service
echo ==============================================

python -m src.rag_service.rag_server
pause