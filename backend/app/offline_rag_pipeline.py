from typing import List, Dict, Optional
import re

class OfflineRAGPipeline:
    def __init__(self, vector_manager, llm_handler):
        self.vector_manager = vector_manager
        self.llm_handler = llm_handler

    def query(self, query: str, doc_id: str = None) -> str:
        """Simple query processing"""
        try:
            # Get relevant documents
            docs = self.vector_manager.similarity_search(query, k=5, filter_source=doc_id)
            
            if not docs:
                return "I couldn't find relevant information in the document."
            
            # Combine content from documents
            context = ""
            for doc in docs:
                context += doc.page_content + "\n\n"
            
            # Create prompt
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            
            # Generate answer using LLM handler
            answer = self.llm_handler.generate(prompt)
            
            return answer if answer else "I couldn't generate an answer for your question."
            
        except Exception as e:
            print(f"RAG query error: {e}")
            return f"Error processing query: {str(e)}"

    def retrieve(self, query: str, doc_id: str = None, k: int = 5):
        """Simple retrieval"""
        return self.vector_manager.similarity_search(query, k=k, filter_source=doc_id)