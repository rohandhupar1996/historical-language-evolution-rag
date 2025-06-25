# Historical German Language Evolution RAG System

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)
![Pipeline](https://img.shields.io/badge/pipeline-6%20phases-orange.svg)

A comprehensive Retrieval-Augmented Generation (RAG) system for analyzing the evolution of the German language across historical periods using the German Manchester Corpus (GerManC). This project combines traditional Natural Language Processing with modern vector databases and semantic search to enable sophisticated linguistic research.

## 📖 Project Story

The German language has undergone remarkable transformations over the past millennium. From Middle High German texts of the 12th century to modern standardized German, linguistic patterns, vocabulary, and grammatical structures have evolved continuously. Understanding these changes requires analyzing vast amounts of historical texts—a task perfectly suited for modern computational linguistics.

This project was born from the need to create an intelligent system that could:
- **Automatically process** thousands of historical German texts
- **Extract linguistic features** across different time periods and genres
- **Enable semantic search** through centuries of language evolution
- **Provide intelligent answers** about historical language patterns

The result is a production-ready RAG system that transforms raw historical texts into an interactive knowledge base, allowing researchers to ask natural language questions about German language evolution and receive contextually-aware answers backed by primary source material.

## 🎯 Key Features

### 🔍 **Intelligent Semantic Search**
- Vector-based similarity search across historical texts
- Period-specific filtering (1050-2000 CE)
- Genre-aware analysis (Legal, Scientific, Literary, etc.)
- Multi-strategy word evolution tracking

### 🤖 **Question-Answering System**
- Natural language queries about historical German
- Context-aware responses with source citations
- Temporal analysis of linguistic phenomena
- Support for both simple retrieval and LLM-powered generation

### 📊 **Comprehensive Analytics**
- Spelling variant analysis across time periods
- Word frequency evolution tracking
- Linguistic feature extraction and comparison
- Statistical insights into language change patterns

### 🏗️ **Production Architecture**
- Modular 6-phase processing pipeline
- PostgreSQL for structured linguistic data
- ChromaDB for vector embeddings
- FastAPI REST endpoints
- Comprehensive validation and logging

## 🛠️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Texts     │ -> │  GATE Pipeline  │ -> │   PostgreSQL    │
│ (GerManC Corpus)│    │ (NLP Features)  │    │  (Structured)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │ <- │   FastAPI REST  │ <- │    ChromaDB     │
│  (Coming Soon)  │    │      API        │    │ (Vector Store)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
historical-language-evolution-rag/
├── 📂 src/                        # Source code directory
│   ├── 📊 Data Pipeline Modules
│   │   ├── germanc_organizer/     # Phase 1: File organization by period/genre
│   │   ├── gate_preprocessor/     # Phase 2: NLP feature extraction  
│   │   ├── validation_suite/      # Phase 3: Quality assurance
│   │   └── prepare_pipeline/      # Phase 4: Data chunking & preparation
│   │
│   ├── 🗄️ Backend Systems
│   │   ├── access_pipeline/       # Phase 5: PostgreSQL setup & REST API
│   │   └── rag_system/           # Phase 6: Vector DB & semantic search
│   │
│   └── 📝 Execution Scripts
│       ├── organize.py           # Execute Phase 1
│       ├── preprocess.py         # Execute Phase 2
│       ├── validate.py           # Execute Phase 3
│       ├── prepare.py            # Execute Phase 4
│       ├── access.py             # Execute Phase 5
│       └── rag.py                # Execute Phase 6
│
├── 📂 data/                       # Data storage
│   ├── raw_corpus/               # Original GerManC files
│   ├── organized_corpus/         # Phase 1 output
│   ├── preprocessed/             # Phase 2 output  
│   └── prepared/                 # Phase 4 output
│
├── 📂 german_corpus_vectordb/     # ChromaDB vector storage
├── 📂 docs/                       # Documentation
├── 📂 config/                     # Configuration files
├── 📂 tests/                      # Unit and integration tests
├── 📂 utils/                      # Utility scripts
├── 📂 Notebook/                   # Jupyter analysis notebooks
│
└── 🔧 Project Files
    ├── requirements.txt           # Python dependencies
    ├── pyproject.toml            # Project configuration
    ├── setup.py                  # Package setup
    └── README.md                 # This documentation
```

## 🚀 Quick Start

### Prerequisites

```bash
# System requirements
Python 3.9+
PostgreSQL 12+
Git LFS (for large corpus files)

# Required disk space
~10GB for full GerManC corpus
~5GB for processed embeddings
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/historical-language-evolution-rag.git
cd historical-language-evolution-rag
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure PostgreSQL**
```bash
# Create database
createdb germanc_corpus

# Update database config in access_pipeline/config.py
# Set your PostgreSQL credentials
```

4. **Download GerManC Corpus**
```bash
# Place your GerManC corpus files in:
mkdir data/raw_corpus
# Copy .txt files organized by period folders
```

## 📋 Step-by-Step Execution

### Phase 1: File Organization
```bash
cd src
python organize.py ../data/raw_corpus ../data/organized_corpus
```
**What it does:** Sorts historical texts by time period and genre, validates file structure, creates metadata catalogs.

### Phase 2: Linguistic Preprocessing  
```bash
cd src
python preprocess.py ../data/organized_corpus ../data/preprocessed
```
**What it does:** Runs GATE NLP pipeline to extract linguistic features, normalizes historical spelling, performs POS tagging and tokenization.

### Phase 3: Quality Validation
```bash
cd src
python validate.py ../data/preprocessed
```
**What it does:** Validates preprocessing quality, checks feature extraction completeness, generates quality reports.

### Phase 4: Data Preparation
```bash
cd src
python prepare.py ../data/preprocessed ../data/prepared
```
**What it does:** Creates text chunks optimized for retrieval, builds word frequency tables, prepares database import files.

### Phase 5: Database & API Setup
```bash
cd src
python access.py ../data/prepared --start-api
```
**What it does:** Imports data into PostgreSQL, creates optimized indexes, starts REST API server for data access.

### Phase 6: RAG System Deployment
```bash
cd src
python rag.py --test --limit 1000
```
**What it does:** Creates vector embeddings, sets up ChromaDB, enables semantic search and question-answering.

### Configuration

### Database Configuration
```python
# src/access_pipeline/config.py
DEFAULT_DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'germanc_corpus',
    'user': 'your_username',
    'password': 'your_password'
}
```

### Embedding Model Configuration
```python
# src/rag_system/config.py
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_VECTOR_DB_PATH = "./chroma_db"
```

## 🔍 Usage Examples

### Semantic Search
```python
import sys
sys.path.append('src')
from rag_system import GermanRAGPipeline

rag = GermanRAGPipeline(db_config, "./vectordb")
rag.setup_qa_system()

# Search for language patterns
results = rag.semantic_search("mittelalterliche deutsche sprache", k=5)

# Period-specific search
results = rag.semantic_search("rechtliche begriffe", period_filter="1350-1650")
```

### Question Answering
```python
# Ask about language evolution
answer = rag.ask_question("Wie entwickelte sich die deutsche Rechtsprache im Mittelalter?")
print(answer['answer'])
print("Sources:", [doc['metadata'] for doc in answer['source_documents']])
```

### Language Evolution Analysis
```python
# Track word changes across periods
evolution = rag.analyze_language_evolution("recht", periods=["1350-1650", "1650-1800"])
for period, data in evolution['periods'].items():
    print(f"{period}: {data['context_count']} contexts found")
```

### REST API Usage
```bash
# Start API server
cd src
python access.py ../data/prepared --start-api

# Query endpoints
curl "http://localhost:8000/search/mittelalterliche%20sprache?period=1350-1650"
curl "http://localhost:8000/evolution/recht/1350-1650/1650-1800"
```

## 📊 Data Schema

### PostgreSQL Tables
- **chunks**: Text segments with metadata (period, genre, word_count)
- **spelling_variants**: Historical spelling variations with normalizations  
- **word_frequencies**: Term frequency across periods and genres
- **linguistic_features**: Extracted grammatical and syntactic features

### Vector Database
- **ChromaDB Collection**: Semantic embeddings of text chunks with metadata
- **Embedding Model**: Multilingual sentence transformer (384 dimensions)
- **Search Index**: Optimized for similarity queries and metadata filtering

## 🧪 Testing

```bash
# Run full test suite
python -m pytest tests/

# Test individual phases (from src directory)
cd src
python validate.py ../data/preprocessed --test-mode
python rag.py --test --limit 100

# Integration tests
python tests/test_full_pipeline.py
```

## 📈 Performance Metrics

- **Corpus Size**: ~50,000 historical German texts
- **Time Coverage**: 1050-2000 CE (950 years)  
- **Processing Speed**: ~500 texts/minute (Phase 2)
- **Search Latency**: <100ms for semantic queries
- **Embedding Creation**: ~1000 chunks/minute
- **Database Size**: ~5GB (structured) + ~2GB (vectors)

## 🔬 Research Applications

### Historical Linguistics
- Track phonological changes across centuries
- Analyze syntactic evolution patterns
- Study lexical semantic shifts

### Digital Humanities  
- Explore genre-specific language patterns
- Investigate sociolinguistic variations
- Support comparative historical analysis

### Computational Linguistics
- Benchmark historical NLP tools
- Develop diachronic language models
- Test cross-temporal retrieval systems

## 🛣️ Roadmap

### Phase 7: Web Interface (In Progress)
- [ ] Interactive web dashboard
- [ ] Visual timeline of language changes  
- [ ] Advanced query builder
- [ ] Export functionality for research data

### Future Enhancements
- [ ] Multi-language support (Historical English, Latin)
- [ ] Advanced temporal modeling
- [ ] Integration with linguistic databases
- [ ] Machine learning-based change detection
- [ ] API versioning and rate limiting

## 🤝 Contributing

We welcome contributions from linguists, historians, and developers!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Use type hints in Python code
- Write descriptive commit messages

## 📚 Documentation

- [API Documentation](docs/api.md) - REST endpoint specifications
- [Architecture Guide](docs/architecture.md) - System design details
- [Linguistic Features](docs/features.md) - NLP processing overview
- [Database Schema](docs/schema.md) - Data structure documentation
- [Deployment Guide](docs/deployment.md) - Production setup instructions

## 🐛 Troubleshooting

### Common Issues

**"Numpy is not available" error**
```bash
pip install numpy==1.24.3 sentence-transformers==2.7.0
```

**PostgreSQL connection errors**
```bash
# Check PostgreSQL service
sudo service postgresql status
# Update connection config in access_pipeline/config.py
```

**GATE processing failures**
```bash
# Verify Java installation
java -version
# Check GATE installation in gate_preprocessor/
```

### Getting Help
- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/historical-language-evolution-rag/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/historical-language-evolution-rag/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GerManC Corpus Team** - For providing the historical German text corpus
- **GATE Development Team** - For the robust NLP processing framework  
- **Sentence Transformers** - For multilingual embedding models
- **ChromaDB Team** - For the efficient vector database
- **Digital Humanities Community** - For inspiration and research collaboration

## 📊 Citation

If you use this system in your research, please cite:

```bibtex
@software{german_rag_system,
  title={Historical German Language Evolution RAG System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/historical-language-evolution-rag},
  note={A comprehensive system for analyzing German language evolution using RAG}
}
```

---

<div align="center">

**Built with ❤️ for Historical Linguistics Research**

[🌟 Star this repo](https://github.com/yourusername/historical-language-evolution-rag) | [🍴 Fork it](https://github.com/yourusername/historical-language-evolution-rag/fork) | [📖 Read the docs](docs/)

</div>