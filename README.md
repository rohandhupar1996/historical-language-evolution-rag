# Historical-Language-Evolution-Rag
Historical Language Evolution RAG AI assistant analyzing German language evolution (1600-1900) using historical texts. Answers questions like "How did German word order change?" with cited evidence from historical documents. Uses TEI-XML processing, PostgreSQL + vector DB, and RAG+GPT-4 to make linguistic research interactive.


# Historical Language Evolution RAG

AI assistant analyzing German language evolution (1600-1900) using historical texts and RAG technology.

## Overview
This system processes historical German texts from archives, extracts linguistic patterns, and provides an AI-powered interface for researchers to query language evolution with cited evidence.

## Features
- 🗂️ **Data Collection**: Automated download from DTA and CLARIN
- ⚙️ **XML Processing**: TEI-P5 validation and XSLT transformations  
- 🗃️ **Database Storage**: PostgreSQL + ChromaDB for temporal data
- 🤖 **RAG System**: GPT-4 powered linguistic analysis
- 🖥️ **Web Interface**: Interactive dashboard with visualizations

## Quick Start
```bash
# Setup
git clone <repository>
cd historical-language-evolution-rag
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys and database credentials

# Run
python main.py --collect  # Download corpus
python main.py --process  # Process XML data
python main.py --serve    # Start web interface
```

## Architecture
```
├── src/
│   ├── data_collection/    # DTA/CLARIN downloaders
│   ├── xml_processing/     # TEI processing & XSLT
│   ├── database/          # PostgreSQL & ChromaDB managers
│   ├── rag/               # RAG pipeline & LLM interface
│   ├── api/               # REST API endpoints
│   └── web/               # Web interface
├── data/                  # Corpus data & processed files
├── config/                # Configuration files
└── docs/                  # Documentation
```

## Technologies
- **XML**: TEI-P5, XSLT 3.0, RelaxNG, XPath
- **Databases**: PostgreSQL, ChromaDB  
- **AI**: OpenAI GPT-4, LangChain, Sentence Transformers
- **Web**: FastAPI, React, D3.js

## License
Academic use only - see LICENSE file