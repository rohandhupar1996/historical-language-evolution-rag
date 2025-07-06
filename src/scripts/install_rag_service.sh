# ==========================================
# FILE: install_rag_service.sh
# ==========================================
#!/bin/bash

echo "🚀 Installing RAG Service Dependencies..."
echo "========================================"

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ $(echo "$python_version < 3.9" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.9+ required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version OK: $python_version"

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements_rag.txt

# Check if critical packages installed correctly
echo "🔍 Checking critical installations..."

if python -c "import chromadb" 2>/dev/null; then
    echo "✅ ChromaDB installed"
else
    echo "❌ ChromaDB installation failed"
    exit 1
fi

if python -c "import sentence_transformers" 2>/dev/null; then
    echo "✅ Sentence Transformers installed"
else
    echo "❌ Sentence Transformers installation failed"
    exit 1
fi

if python -c "import fastapi" 2>/dev/null; then
    echo "✅ FastAPI installed"
else
    echo "❌ FastAPI installation failed"
    exit 1
fi

echo ""
echo "🎉 RAG Service dependencies installed successfully!"
echo "📝 Next steps:"
echo "   1. Ensure vector database exists: python rag.py --test --limit 1000"
echo "   2. Start RAG service: ./start_rag_service.sh"
echo "   3. Start Streamlit: streamlit run streamlit_app.py"