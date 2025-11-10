# Patient Discharge Assistant

AI-powered medical document analysis system with MCP QueryHandler protocol for forensic-level document extraction and intelligent query processing.

## Features

- **Document Upload & Processing**: Supports PDF, DOCX, TXT, images
- **Real-time Chat Interface**: Interactive Q&A about medical documents
- **MCP Protocol**: EXTRACT and ANSWER modes for comprehensive analysis
- **Tokenization & Embeddings**: Accurate text tokenization and semantic embeddings using transformers
- **Semantic Search**: AI-powered similarity search with 384-dimensional embeddings
- **Section-based Navigation**: Browse document sections when searches fail
- **Comprehensive Data Extraction**: Patient info, diagnosis, medications, instructions

## Quick Start

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

### Run Application
```bash
# Start backend
python start_server.py

# Start frontend (new terminal)
cd frontend
npm start
```

## API Endpoints

### Document Processing
- `POST /upload` - Upload document (EXTRACT mode)
- `POST /query` - Query document (ANSWER mode)
- `POST /mcp-extract` - Pure MCP EXTRACT
- `POST /mcp-answer` - Pure MCP ANSWER

### Embeddings & Search
- `POST /tokenize` - Tokenize text using transformers
- `POST /generate-embedding` - Generate embedding vector
- `POST /semantic-search` - Semantic similarity search
- `GET /embeddings-stats` - Get embeddings statistics

### System
- `GET /health` - Health check
- `GET /supported-formats` - Supported file formats

## System Status
```bash
python system_status.py
```

## Architecture

- **Backend**: FastAPI with MCP QueryHandler
- **Frontend**: React with real-time chat
- **Document Processing**: ForensicDocumentExtractor
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **Tokenization**: Hugging Face Transformers
- **Semantic Search**: Cosine similarity with 384-dim vectors
- **Query System**: Pattern-based extraction with semantic fallbacks

## License

MIT License