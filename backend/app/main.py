from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os
import asyncio
import signal
import sys
from .offline_document_parser import ForensicDocumentExtractor
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

# Initialize forensic extractor
forensic_extractor = ForensicDocumentExtractor()
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
query_manager = OfflineQueryManager(forensic_extractor, llm_handler, rag_pipeline)

# Store processing status
processing_status = {}

async def process_document_async(file_path: str, filename: str):
    """Process document in background using forensic extraction"""
    try:
        print(f"🔄 Starting forensic extraction for: {filename}")
        
        # EXTRACT mode: Forensic-level document extraction
        extraction_result = forensic_extractor.extract_document(file_path)
        
        if extraction_result.get('extraction_remarks'):
            for remark in extraction_result['extraction_remarks']:
                if remark.get('issue') == 'extraction_error':
                    processing_status[filename] = {
                        "status": "error", 
                        "message": remark.get('note', 'Extraction failed')
                    }
                    return
        
        text = extraction_result.get('full_text', '')
        print(f"✅ Forensic extraction completed: {len(text)} characters")
        
        # Add to vector store
        vector_manager.add_document(filename, text)
        
        # Process with query manager for backward compatibility
        query_manager.process_document(text, filename)
        
        processing_status[filename] = {
            "status": "completed",
            "message": f"Forensic extraction completed. Extracted {len(text)} characters",
            "filename": filename,
            "extracted_data": query_manager.structured_data,
            "forensic_data": extraction_result
        }
        
        print(f"✅ Forensic processing completed for: {filename}")
        
    except asyncio.CancelledError:
        print(f"⚠️ Processing cancelled for: {filename}")
        processing_status[filename] = {"status": "cancelled", "message": "Processing was cancelled"}
        raise
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

        "extracted_data": status.get("extracted_data", {}),
        "forensic_data": status.get("forensic_data", {})
    }

@app.post("/query")
async def query_document(payload: dict):
    """Query using MCP QueryHandler protocol"""
    query = payload.get("query", "").strip()
    filename = payload.get("filename", "")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        # ANSWER mode: Query against extracted forensic data
        mcp_response = forensic_extractor.answer_query(query)
        
        # Also get traditional response for compatibility
        traditional_result = query_manager.handle_query(query)
        
        # Add section selector flag if no good match found
        show_sections = False
        if "not found" in traditional_result["answer"].lower() or traditional_result.get("confidence", 0) < 0.5:
            show_sections = True
        
        return {
            "mcp_response": mcp_response,
            "answer": traditional_result["answer"],
            "confidence": traditional_result.get("confidence", 0.8),
            "category": traditional_result.get("category", "General"),
            "suggestions": traditional_result.get("suggestions", []),
            "medical_instructions": traditional_result.get("medical_instructions", []),
            "safety_alerts": traditional_result.get("safety_alerts", []),
            "metadata": traditional_result.get("mcp_metadata", {}),
            "entities": traditional_result.get("entities", []),
            "extracted_sections": len(traditional_result.get("extracted_data", {})),
            "data_quality": "complete" if traditional_result.get("extracted_data") else "limited",
            "response_type": "forensic_mcp_enhanced",
            "template_used": "mcp_queryhandler",
            "show_sections": show_sections
        }
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/extract-data")
async def extract_document_data():
    """Extract forensic-level structured data from document"""
    try:
        if not forensic_extractor.extracted_data:
            raise HTTPException(status_code=400, detail="No document data available. Please upload a document first.")
        
        return {
            "status": "success", 
            "forensic_data": forensic_extractor.extracted_data,
            "traditional_data": query_manager.structured_data,
            "extraction_type": "forensic_mcp",
            "blocks_count": len(forensic_extractor.extracted_data.get('blocks', [])),
            "tables_count": len(forensic_extractor.extracted_data.get('tables', [])),
            "forms_count": len(forensic_extractor.extracted_data.get('forms', [])),
            "entities_count": len(forensic_extractor.extracted_data.get('entities', []))
        }
    except Exception as e:
        print(f"Extract data error: {e}")
        raise HTTPException(status_code=500, detail=f"Data extraction failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Patient Discharge Assistant API running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mcp_protocol": "active",
        "forensic_extractor": "operational",
        "components": {
            "extract_mode": True,
            "answer_mode": True,
            "legacy_compatibility": True
        }
    }

# Graceful shutdown handler
def signal_handler(signum, frame):
    print("\n🔄 Gracefully shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.get("/supported-formats")
async def get_supported_formats():
    """Return supported file formats for forensic extraction"""
    return {
        "supported_formats": ['.pdf', '.txt', '.docx', '.doc', '.png', '.jpg', '.jpeg', '.xlsx', '.pptx', '.html', '.csv', '.eml'],
        "extraction_capabilities": ["text", "tables", "forms", "metadata", "entities", "blocks", "chunks"],
        "protocol": "MCP QueryHandler",
        "modes": ["EXTRACT", "ANSWER"]
    }

@app.post("/mcp-extract")
async def mcp_extract_only(file_path: str):
    """Pure MCP EXTRACT mode endpoint"""
    try:
        extraction_result = forensic_extractor.extract_document(file_path)
        return extraction_result
    except asyncio.CancelledError:
        raise HTTPException(status_code=499, detail="Request cancelled")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP extraction failed: {str(e)}")

@app.post("/mcp-answer")
async def mcp_answer_only(payload: dict):
    """Pure MCP ANSWER mode endpoint"""
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        mcp_response = forensic_extractor.answer_query(query)
        return {"mcp_response": mcp_response}
    except asyncio.CancelledError:
        raise HTTPException(status_code=499, detail="Request cancelled")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP answer failed: {str(e)}")

@app.get("/available-sections")
async def get_available_sections():
    """Get all available sections for UI selection"""
    try:
        if not query_manager.structured_data:
            return {"sections": [], "message": "No document uploaded"}
        
        sections = []
        for key, value in query_manager.structured_data.items():
            if value and isinstance(value, str):
                section_name = key.replace('_', ' ').title()
                preview = value[:80] + "..." if len(value) > 80 else value
                sections.append({
                    "key": key,
                    "name": section_name,
                    "preview": preview,
                    "length": len(value)
                })
        
        return {"sections": sections, "total_sections": len(sections)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sections: {str(e)}")

@app.post("/get-section")
async def get_specific_section(payload: dict):
    """Get specific section content by key"""
    section_key = payload.get("section_key", "").strip()
    if not section_key:
        raise HTTPException(status_code=400, detail="Section key is required")
    
    try:
        if not query_manager.structured_data:
            raise HTTPException(status_code=400, detail="No document uploaded")
        
        content = query_manager.structured_data.get(section_key, "")
        if not content:
            raise HTTPException(status_code=404, detail="Section not found")
        
        section_name = section_key.replace('_', ' ').title()
        
        # Format as section response
        answer = f"📄 **{section_name}**\n\n{content}\n\n📝 This is the complete content from the selected section."
        
        return {
            "answer": answer,
            "category": "get_section",
            "confidence": 0.95,
            "suggestions": [],
            "section_name": section_name,
            "section_key": section_key
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get section: {str(e)}")