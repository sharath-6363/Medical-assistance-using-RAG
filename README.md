# Patient Discharge Assistant

AI-powered medical document analysis system with MCP QueryHandler protocol for forensic-level document extraction and intelligent query processing.

## Features

- **Document Upload & Processing**: Supports PDF, DOCX, TXT, images
- **Real-time Chat Interface**: Interactive Q&A about medical documents
- **MCP Protocol**: EXTRACT and ANSWER modes for comprehensive analysis
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

- `POST /upload` - Upload document (EXTRACT mode)
- `POST /query` - Query document (ANSWER mode)
- `GET /health` - Health check
- `POST /mcp-extract` - Pure MCP EXTRACT
- `POST /mcp-answer` - Pure MCP ANSWER

## System Status
```bash
python system_status.py
```

## Architecture

- **Backend**: FastAPI with MCP QueryHandler
- **Frontend**: React with real-time chat
- **Document Processing**: ForensicDocumentExtractor
- **Query System**: Pattern-based extraction with fallbacks

## License

MIT License