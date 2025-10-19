from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
from app.offline_document_parser import OfflineDocumentParser
from app.offline_llm_handler import OfflineLLMHandler
from app.offline_query_manager import OfflineQueryManager
from app.offline_rag_pipeline import OfflineRAGPipeline

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Patient Discharge Assistant - Offline")

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
        from langchain.schema import Document
        results = []
        for doc_id, text in self.documents.items():
            if filter_source and doc_id != filter_source:
                continue
            if any(word.lower() in text.lower() for word in query.split()):
                results.append(Document(page_content=text[:500], metadata={"source": doc_id}))
        return results[:k]
    
    def get_document_text(self, doc_id):
        return self.documents.get(doc_id, "")

vector_manager = SimpleVectorManager()
rag_pipeline = OfflineRAGPipeline(vector_manager, llm_handler)
query_manager = OfflineQueryManager(document_parser, llm_handler, rag_pipeline)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process medical document"""
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        supported_extensions = ['.pdf', '.txt', '.docx', '.doc', '.png', '.jpg', '.jpeg']
        
        if file_ext not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}"
            )
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract and process text
        text = document_parser.extract_text(str(file_path))
        
        # Add to vector store
        vector_manager.add_document(file.filename, text)
        
        # Process document
        query_manager.process_document(text, file.filename)
        
        return {
            "status": "processing",
            "filename": file.filename,
            "message": "Document processed successfully",
            "processing_id": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_document(payload: dict):
    """Query the uploaded document"""
    query = payload.get("query", "").strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        result = query_manager.handle_query(query)
        return {
            "answer": result["answer"],
            "confidence": result.get("confidence", 0.8),
            "category": result.get("category", "General")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processing-status/{filename}")
async def get_processing_status(filename: str):
    """Check processing status"""
    return {
        "filename": filename,
        "status": "completed",
        "message": "Document processed successfully",
        "extracted_data": query_manager.structured_data
    }

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
    return {"message": "Patient Discharge Assistant - Offline Mode"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)