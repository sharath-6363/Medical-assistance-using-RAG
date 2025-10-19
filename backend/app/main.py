from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os
import asyncio
from .offline_document_parser import OfflineDocumentParser
from .offline_llm_handler import OfflineLLMHandler
from .offline_query_manager import OfflineQueryManager
from .offline_rag_pipeline import OfflineRAGPipeline

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Patient Discharge Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize offline components
document_parser = OfflineDocumentParser()
llm_handler = OfflineLLMHandler()

# Create a simple vector manager for RAG
class SimpleVectorManager:
    def __init__(self):
        self.documents = {}
    
    def add_document(self, doc_id, text):
        self.documents[doc_id] = text
    
    def similarity_search(self, query, k=5, filter_source=None):
        # Simple document class
        class SimpleDocument:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        
        results = []
        query_lower = query.lower()
        
        for doc_id, text in self.documents.items():
            if filter_source and doc_id != filter_source:
                continue
            
            # Find relevant sections
            lines = text.split('\n')
            relevant_lines = []
            
            for line in lines:
                if any(word in line.lower() for word in query_lower.split() if len(word) > 2):
                    relevant_lines.append(line)
            
            if relevant_lines:
                content = '\n'.join(relevant_lines[:10])  # Top 10 relevant lines
                results.append(SimpleDocument(content, {"source": doc_id}))
        
        return results[:k]
    
    def get_document_text(self, doc_id):
        return self.documents.get(doc_id, "")

vector_manager = SimpleVectorManager()
rag_pipeline = OfflineRAGPipeline(vector_manager, llm_handler)
query_manager = OfflineQueryManager(document_parser, llm_handler, rag_pipeline)

# Store processing status
processing_status = {}

async def process_document_async(file_path: str, filename: str):
    """Process document in background"""
    try:
        print(f"🔄 Starting async processing for: {filename}")
        
        # Extract text using document parser
        print(f"🔍 Extracting text from: {file_path}")
        text = document_parser.extract_text(file_path)
        
        print(f"📊 Extracted text length: {len(text)}")
        print(f"📊 Text preview: {text[:200]}...")
        
        if not text.strip():
            processing_status[filename] = {
                "status": "error", 
                "message": "Could not extract any text from the document"
            }
            return
        
        print(f"✅ Text extraction completed: {len(text)} characters")
        
        # Add to vector store
        vector_manager.add_document(filename, text)
        
        # Process document with offline parser
        print(f"🔄 Processing document with query manager...")
        query_manager.process_document(text, filename)
        print(f"📊 Query manager structured_data: {len(query_manager.structured_data)} sections")
        
        processing_status[filename] = {
            "status": "completed",
            "message": f"Document processed successfully. Extracted {len(text)} characters",
            "filename": filename,
            "extracted_data": query_manager.structured_data
        }
        
        print(f"✅ Document processing completed for: {filename}")
        
    except Exception as e:
        print(f"❌ Processing error for {filename}: {e}")
        processing_status[filename] = {
            "status": "error",
            "message": f"Processing failed: {str(e)}"
        }

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process medical document"""
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        supported_extensions = ['.pdf', '.txt', '.docx', '.doc', '.png', '.jpg', '.jpeg']
        
        if file_ext not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(supported_extensions)}"
            )
        
        # Validate file size (50MB limit)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum size is 50MB."
            )
        
        print(f"📁 Processing file: {file.filename} ({file_size} bytes)")
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"✅ File saved: {file_path}")
        
        # Initialize processing status
        processing_status[file.filename] = {
            "status": "processing",
            "message": "Starting document processing..."
        }
        
        # Process in background
        background_tasks.add_task(process_document_async, str(file_path), file.filename)
        
        return {
            "status": "processing",
            "filename": file.filename,
            "message": "Document upload successful. Processing started...",
            "processing_id": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/processing-status/{filename}")
async def get_processing_status(filename: str):
    """Check processing status"""
    status = processing_status.get(filename, {})
    return {
        "filename": filename,
        "status": status.get("status", "unknown"),
        "message": status.get("message", "Status not available"),

        "extracted_data": status.get("extracted_data", {})
    }

@app.post("/query")
async def query_document(payload: dict):
    """Query the uploaded document"""
    query = payload.get("query", "").strip()
    filename = payload.get("filename", "")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        # MCP protocol integrated in query manager
        result = query_manager.handle_query(query)
        return {
            "answer": result["answer"],
            "confidence": result.get("confidence", 0.8),
            "category": result.get("category", "General"),
            "suggestions": result.get("suggestions", []),
            "metadata": result.get("mcp_metadata", {}),
            "entities": result.get("entities", []),
            "extracted_sections": len(result.get("extracted_data", {})),
            "data_quality": "complete" if result.get("extracted_data") else "limited"
        }
    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/extract-data")
async def extract_document_data():
    """Extract all structured data from document"""
    try:
        if not query_manager.structured_data:
            raise HTTPException(status_code=400, detail="No document data available. Please upload a document first.")
        
        return {
            "status": "success", 
            "data": query_manager.structured_data,
            "sections": list(query_manager.structured_data.keys()) if query_manager.structured_data else []
        }
    except Exception as e:
        print(f"Extract data error: {e}")
        raise HTTPException(status_code=500, detail=f"Data extraction failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Patient Discharge Assistant API running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/supported-formats")
async def get_supported_formats():
    """Return supported file formats"""
    return {
        "supported_formats": ['.pdf', '.txt', '.docx', '.doc', '.png', '.jpg', '.jpeg']
    }