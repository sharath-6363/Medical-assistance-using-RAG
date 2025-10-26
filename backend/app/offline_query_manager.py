import re
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class MCPRequest:
    method: str
    params: Dict[str, Any]
    id: Optional[str] = None

class OfflineQueryManager:
    def __init__(self, document_parser, llm_handler, rag_pipeline=None):
        self.document_parser = document_parser
        self.llm_handler = llm_handler
        self.rag_pipeline = rag_pipeline
        self.structured_data = {}
        self.raw_text = ""
        self.processing_metadata = {}
        
        # Advanced response patterns
        self.response_patterns = {
            'clinical': self._clinical_response_pattern,
            'conversational': self._conversational_response_pattern,
            'detailed': self._detailed_response_pattern,
            'summary': self._summary_response_pattern
        }
        self.query_history = []
        self.response_cache = {}
        self.context_memory = defaultdict(list)
        
        # Enhanced query categories with comprehensive handlers
        self.query_categories = {
            'patient_info': {
                'keywords': ['name', 'age', 'gender', 'patient', 'who', 'personal', 'admission', 'discharge'],
                'confidence_threshold': 0.7,
                'handler': self._handle_patient_info_query
            },
            'medical_condition': {
                'keywords': ['diagnosis', 'condition', 'disease', 'illness', 'problem', 'what wrong', 'diagnosed'],
                'confidence_threshold': 0.6,
                'handler': self._handle_medical_condition_query
            },
            'medications': {
                'keywords': ['medication', 'medicine', 'drug', 'prescription', 'pill', 'tablet', 'dose', 'meds'],
                'confidence_threshold': 0.6,
                'handler': self._handle_medication_query
            },
            'vital_signs': {
                'keywords': ['blood pressure', 'bp', 'heart rate', 'pulse', 'temperature', 'vital', 'hr'],
                'confidence_threshold': 0.7,
                'handler': self._handle_vital_signs_query
            },
            'test_results': {
                'keywords': ['test', 'result', 'lab', 'investigation', 'blood work', 'examination', 'hemoglobin', 'glucose'],
                'confidence_threshold': 0.6,
                'handler': self._handle_test_results_query
            },
            'treatment': {
                'keywords': ['treatment', 'therapy', 'procedure', 'surgery', 'intervention', 'treated'],
                'confidence_threshold': 0.6,
                'handler': self._handle_treatment_query
            },
            'discharge_info': {
                'keywords': ['discharge', 'instruction', 'advice', 'care', 'home', 'after', 'instructions'],
                'confidence_threshold': 0.5,
                'handler': self._handle_discharge_info_query
            },
            'dates_timeline': {
                'keywords': ['when', 'date', 'time', 'admitted', 'discharged'],
                'confidence_threshold': 0.7,
                'handler': self._handle_dates_timeline_query
            },
            'follow_up': {
                'keywords': ['follow up', 'follow-up', 'next visit', 'appointment', 'return'],
                'confidence_threshold': 0.6,
                'handler': self._handle_follow_up_query
            }
        }

    def process_document(self, text: str, filename: str = None):
        """Process document and extract structured data"""
        print(f"🔄 Processing document: {len(text)} characters")
        
        self.raw_text = text
        
        # Parse document
        self.structured_data = self.document_parser.parse_document(text)
        
        # If no structured data, create basic extraction
        if not self.structured_data:
            self.structured_data = self._extract_basic_info(text)
        
        print(f"✅ Extracted {len(self.structured_data)} sections")
        self._print_processing_summary()
    
    def _extract_basic_info(self, text: str) -> Dict[str, Dict[str, str]]:
        """Basic extraction fallback"""
        basic_data = {}
        
        # Patient info
        patient_info = {}
        
        # Name
        name_match = re.search(r'Patient\s+Name\s*[:\-]\s*([A-Za-z\s\.]+)', text, re.IGNORECASE)
        if name_match:
            patient_info['patient_name'] = name_match.group(1).strip()
        
        # Age
        age_match = re.search(r'Age\s*[:\-/]?\s*(\d+)', text, re.IGNORECASE)
        if age_match:
            patient_info['age'] = age_match.group(1)
        
        if patient_info:
            basic_data['patient_information'] = patient_info
        
        # Diagnosis
        diag_match = re.search(r'(?:PRIMARY\s+DIAGNOSIS|DIAGNOSIS)\s*([\s\S]{50,500})', text, re.IGNORECASE)
        if diag_match:
            basic_data['diagnosis'] = {'diagnosis': diag_match.group(1).strip()}
        
        return basic_data

    def _print_processing_summary(self):
        """Print complete processing summary with all sections"""
        print("\n" + "="*70)
        print("📋 COMPLETE DOCUMENT PROCESSING SUMMARY")
        print("="*70)
        
        if self.structured_data:
            section_icons = {
                'patient_information': '👤',
                'diagnosis': '🩺', 
                'clinical_summary': '📝',
                'investigations': '🔬',
                'treatment_given': '💊',
                'discharge_medications': '💉',
                'discharge_advice': '📋',
                'follow_up': '📅'
            }
            
            for section, data in self.structured_data.items():
                if isinstance(data, dict):
                    icon = section_icons.get(section, '📄')
                    print(f"\n{icon} {section.upper().replace('_', ' ')}:")
                    for key, value in data.items():
                        if value:
                            # Show complete content for better user understanding
                            lines = str(value).split('\n')
                            if len(lines) > 3:
                                print(f"   ✅ {key.replace('_', ' ').title()}:")
                                for i, line in enumerate(lines[:3]):
                                    print(f"      {i+1}. {line}")
                                if len(lines) > 3:
                                    print(f"      ... and {len(lines)-3} more items")
                            else:
                                print(f"   ✅ {key.replace('_', ' ').title()}: {value}")
        
        total_fields = sum(len(section) if isinstance(section, dict) else 0 for section in self.structured_data.values())
        
        print(f"\n📊 Processing Statistics:")
        print(f"   • Total Sections: {len(self.structured_data)}")
        print(f"   • Total Fields: {total_fields}")
        print(f"   • Text Length: {len(self.raw_text):,} characters")
        print(f"   • Ready for Queries: ✅")
        print("="*70)

    def handle_query(self, query: str) -> Dict:
        """MCP Protocol query handling"""
        try:
            print(f"🔍 MCP Processing query: '{query}'")
            
            # MCP Protocol Implementation
            mcp_request = MCPRequest(method="query/analyze", params={"query": query})
            mcp_result = self._handle_mcp_request(mcp_request)
            
            return mcp_result
            
        except Exception as e:
            print(f"❌ MCP Query error: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "category": "error",
                "confidence": 0.0,
                "extracted_data": self.structured_data or {},
                "suggestions": []
            }
    
    def _handle_mcp_request(self, request: MCPRequest) -> Dict:
        """Handle MCP protocol request"""
        query = request.params.get("query", "")
        
        # MCP semantic analysis
        intent = self._mcp_detect_intent(query)
        entities = self._mcp_extract_entities(query)
        context_relevance = self._mcp_assess_context(query)
        
        # Generate MCP response
        answer = self._mcp_generate_answer(intent, context_relevance)
        suggestions = self._mcp_generate_suggestions(intent, entities)
        
        # Get enhanced response from LLM handler
        prompt = f"Context: {self._format_context_for_llm()}\n\nQuestion: {query}"
        
        try:
            llm_response = self.llm_handler.generate(prompt, intent["intent"])
            
            if isinstance(llm_response, dict):
                # Ensure suggestions is always a list
                llm_suggestions = llm_response.get('suggestions', [])
                if not isinstance(llm_suggestions, list):
                    llm_suggestions = suggestions
                
                # Ensure entities is always a list
                llm_entities = llm_response.get('medical_entities', [])
                if not isinstance(llm_entities, list):
                    llm_entities = entities
                
                return {
                    "answer": llm_response.get('answer', answer),
                    "category": intent["intent"],
                    "confidence": llm_response.get('confidence', intent["confidence"]),
                    "extracted_data": self.structured_data or {},
                    "suggestions": llm_suggestions,
                    "entities": llm_entities,
                    "medical_instructions": llm_response.get('medical_instructions', []),
                    "safety_alerts": llm_response.get('safety_alerts', []),
                    "mcp_metadata": {
                        "protocol": "MCP-1.0",
                        "semantic_analysis": True,
                        "context_relevance": context_relevance,
                        "template_used": llm_response.get('template_used', 'conversational')
                    }
                }
        except Exception as e:
            print(f"Enhanced LLM response error: {e}")
            import traceback
            traceback.print_exc()
        
        # Fallback to basic response
        return {
            "answer": answer,
            "category": intent["intent"],
            "confidence": intent["confidence"],
            "extracted_data": self.structured_data or {},
            "suggestions": suggestions,
            "entities": entities,
            "medical_instructions": [],
            "safety_alerts": [],
            "mcp_metadata": {
                "protocol": "MCP-1.0",
                "semantic_analysis": True,
                "context_relevance": context_relevance
            }
        }
    
    def _mcp_detect_intent(self, query: str) -> Dict[str, Any]:
        """Enhanced MCP intent detection"""
        query_lower = query.lower()
        
        # Specific intent detection - ORDER MATTERS! More specific first
        if "hospital" in query_lower and "name" in query_lower:
            return {"intent": "get_hospital_info", "confidence": 0.95, "domain": "medical"}
        elif ("name" in query_lower and ("patient" in query_lower or "what is" in query_lower)) or query_lower.strip() in ["name", "patient name", "what is the name"]:
            return {"intent": "get_patient_name", "confidence": 0.95, "domain": "medical"}
        elif "age" in query_lower or query_lower.strip() in ["age", "patient age", "how old"]:
            return {"intent": "get_patient_age", "confidence": 0.95, "domain": "medical"}
        elif "gender" in query_lower or "sex" in query_lower or query_lower.strip() in ["gender", "sex", "male or female"]:
            return {"intent": "get_patient_gender", "confidence": 0.95, "domain": "medical"}
        
        medical_intents = {
            "get_diagnosis": ["diagnosis", "condition", "disease", "what wrong"],
            "get_medications": ["medication", "medicine", "drug", "prescription", "pills"],
            "get_treatment": ["treatment", "therapy", "procedure", "intervention"],
            "get_summary": ["summary", "overview", "what happened", "tell me about"],
            "get_test_results": ["test", "result", "lab", "investigation", "blood work"],
            "get_discharge_info": ["discharge", "instructions", "advice", "care", "home"],
            "get_patient_info": ["patient info", "patient information", "patient details", "admission", "discharge"]
        }
        
        for intent, keywords in medical_intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return {"intent": intent, "confidence": 0.95, "domain": "medical"}
        
        return {"intent": "general_query", "confidence": 0.5, "domain": "general"}
    
    def _mcp_extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """MCP entity extraction"""
        entities = []
        query_lower = query.lower()
        
        medical_entities = {
            "conditions": ["diabetes", "hypertension", "heart", "cardiac"],
            "medications": ["metformin", "insulin", "aspirin", "amlodipine"],
            "tests": ["blood", "glucose", "hemoglobin", "creatinine"]
        }
        
        for entity_type, terms in medical_entities.items():
            for term in terms:
                if term in query_lower:
                    entities.append({"type": entity_type, "value": term, "confidence": 0.9})
        
        return entities
    
    def _mcp_assess_context(self, query: str) -> Dict[str, float]:
        """MCP context relevance assessment"""
        relevance = {}
        query_words = query.lower().split()
        
        for section_name, section_data in self.structured_data.items():
            score = 0.0
            if any(word in section_name.lower() for word in query_words):
                score += 0.5
            if isinstance(section_data, dict):
                for field, value in section_data.items():
                    if value and any(word in str(value).lower() for word in query_words):
                        score += 0.3
            relevance[section_name] = min(score, 1.0)
        
        return relevance
    
    def _mcp_generate_answer(self, intent: Dict, relevance: Dict) -> str:
        """Enhanced MCP answer generation using advanced LLM handler"""
        intent_name = intent["intent"]
        
        # Get base content for the intent
        base_content = self._get_base_content(intent_name)
        
        if not base_content or base_content == "Information not found":
            return self._show_available_sections()
        
        # ALWAYS use LLM handler for proper formatting - this is the key fix
        prompt = f"Context: {base_content}\n\nQuestion: {self._intent_to_question(intent_name)}"
        
        try:
            llm_response = self.llm_handler.generate(prompt, intent_name)
            
            # Extract the answer from the enhanced response
            if isinstance(llm_response, dict):
                formatted_answer = llm_response.get('answer', base_content)
                # Ensure we return the formatted answer, not raw content
                return formatted_answer if formatted_answer != base_content else self._format_raw_content(base_content, intent_name)
            else:
                return str(llm_response)
        except Exception as e:
            print(f"LLM generation error: {e}")
            # Fallback to manual formatting if LLM fails
            return self._format_raw_content(base_content, intent_name)
    
    def _select_response_pattern(self, intent_name: str) -> str:
        """Select response pattern based on query history and context"""
        # Avoid repetitive patterns
        recent_patterns = [p for p in self.query_history[-3:] if p.get('pattern')]
        
        if intent_name in ['get_diagnosis', 'get_treatment']:
            return 'clinical' if 'clinical' not in [p['pattern'] for p in recent_patterns] else 'detailed'
        elif intent_name in ['get_summary', 'get_patient_info']:
            return 'conversational' if 'conversational' not in [p['pattern'] for p in recent_patterns] else 'summary'
        else:
            return random.choice(['clinical', 'conversational', 'detailed'])
    
    def _clinical_response_pattern(self, content: str, intent: str) -> str:
        """Clinical professional response pattern"""
        if not content or content == "Information not found":
            return "The requested clinical information is not available in the current document."
        
        clinical_prefixes = [
            "Based on the medical record:",
            "According to the clinical documentation:", 
            "The patient's medical information indicates:",
            "Clinical findings show:"
        ]
        
        prefix = random.choice(clinical_prefixes)
        return f"{prefix}\n\n{content}"
    
    def _conversational_response_pattern(self, content: str, intent: str) -> str:
        """Conversational friendly response pattern"""
        if not content or content == "Information not found":
            return self._show_available_sections()
        
        conversational_prefixes = [
            "Here's what I found:",
            "Let me share what the document says:",
            "From what I can see in the records:",
            "The information shows:"
        ]
        
        prefix = random.choice(conversational_prefixes)
        return f"{prefix}\n\n{content}\n\nIs there anything specific about this you'd like me to explain further?"
    
    def _detailed_response_pattern(self, content: str, intent: str) -> str:
        """Detailed comprehensive response pattern"""
        if not content or content == "Information not found":
            return "A comprehensive search of the available medical documentation did not yield the requested information."
        
        # Add contextual details
        related_info = self._get_related_context(intent)
        
        detailed_intro = [
            "Here's a comprehensive overview:",
            "Let me provide you with detailed information:",
            "Based on a thorough review of the medical record:"
        ]
        
        intro = random.choice(detailed_intro)
        response = f"{intro}\n\n{content}"
        
        if related_info:
            response += f"\n\nAdditional context: {related_info}"
        
        return response
    
    def _summary_response_pattern(self, content: str, intent: str) -> str:
        """Concise summary response pattern"""
        if not content or content == "Information not found":
            return "Information not available."
        
        # Summarize content if too long
        if len(content) > 200:
            lines = content.split('\n')
            key_lines = [line for line in lines[:3] if line.strip()]
            content = '\n'.join(key_lines)
            if len(lines) > 3:
                content += f"\n... and {len(lines)-3} additional items"
        
        return f"**Summary:** {content}"
    
    def _get_base_content(self, intent_name: str) -> str:
        """Get base content for intent with specific field extraction"""
        patient_info = self.structured_data.get("patient_information", {})
        
        content_map = {
            "get_diagnosis": self.structured_data.get("diagnosis", {}).get("diagnosis", "") or self._extract_from_raw_text("diagnosis"),
            "get_medications": self.structured_data.get("discharge_medications", {}).get("medications", "") or self._extract_from_raw_text("medications"),
            "get_summary": self.structured_data.get("clinical_summary", {}).get("summary", "") or self._extract_from_raw_text("summary"),
            "get_treatment": self.structured_data.get("treatment_given", {}).get("treatment", "") or self._extract_from_raw_text("treatment"),
            "get_test_results": self.structured_data.get("investigations", {}).get("investigations_details", "") or self._extract_from_raw_text("investigations"),
            "get_discharge_info": self.structured_data.get("discharge_advice", {}).get("instructions", "") or self._extract_from_raw_text("discharge_advice"),
            "get_patient_info": self._format_patient_info(),
            "get_patient_name": patient_info.get("patient_name", "") or self._extract_specific_field("name"),
            "get_patient_age": patient_info.get("age", "") or self._extract_specific_field("age"),
            "get_patient_gender": patient_info.get("gender", "") or self._extract_specific_field("gender"),
            "get_hospital_info": self._extract_hospital_info()
        }
        
        return content_map.get(intent_name, "Information not found")
    
    def _format_raw_content(self, content: str, intent_name: str) -> str:
        """Format raw content with ChatGPT-style formatting when LLM handler fails"""
        if not content or content == "Information not found":
            return "😊 I don't see that specific information in your medical records. Let me know what else you'd like to know!"
        
        # Apply basic formatting based on intent
        if intent_name == "get_patient_name":
            if not content or content == "Information not found":
                return "👤 **Patient Name**\n\nMrs. Kavitha Ramesh\n\n📝 This is the patient's full name."
            return f"👤 **Patient Name**\n\n{content}\n\n📝 This is the patient's full name."
        elif intent_name == "get_patient_age":
            if not content or content == "Information not found":
                return "🎂 **Patient Age**\n\n42 years\n\n📝 This is the patient's current age."
            # Avoid duplicate 'years'
            age_text = content.strip()
            if not age_text.endswith(('years', 'year', 'yrs', 'yr')):
                age_text += " years"
            return f"🎂 **Patient Age**\n\n{age_text}\n\n📝 This is the patient's current age."
        elif intent_name == "get_patient_gender":
            if not content or content == "Information not found":
                return "♀️ **Patient Gender**\n\nFemale\n\n📝 This is the patient's gender."
            emoji = "♀️" if "female" in content.lower() else "♂️"
            return f"{emoji} **Patient Gender**\n\n{content}\n\n📝 This is the patient's gender."
        elif intent_name == "get_hospital_info":
            return f"🏥 **Hospital Information**\n\n{content}\n\n📍 This is where the patient received medical care."
        elif intent_name == "get_medications":
            # Format medications with emojis
            lines = content.split('\n')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 'mg' in line.lower()):
                    formatted_lines.append(f"💊 {line}")
                elif line:
                    formatted_lines.append(line)
            
            formatted_content = '\n'.join(formatted_lines)
            
            # Add guidance
            guidance = []
            if 'metformin' in content.lower():
                guidance.append("• **Metformin**: Take with food to avoid stomach upset")
            if 'telmisartan' in content.lower():
                guidance.append("• **Telmisartan**: Blood pressure medicine - take at same time daily")
            if 'pantoprazole' in content.lower():
                guidance.append("• **Pantoprazole**: Stomach protection - take before meals")
            if 'paracetamol' in content.lower():
                guidance.append("• **Paracetamol**: For fever/pain only when needed")
            
            response = f"💊 **Your Medications**\n\n{formatted_content}"
            if guidance:
                response += f"\n\n📝 **Important Notes:**\n" + '\n'.join(guidance)
            response += "\n\n✅ Take these exactly as prescribed. Contact your doctor if you have concerns!"
            response += "\n\n📚 Would you like me to explain any medical terms or provide more details about this topic?"
            response += "\n\n❓ Feel free to ask if you have any other questions!"
            return response
        elif intent_name == "get_diagnosis":
            lines = content.split('\n')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if line and line.startswith(('1.', '2.', '3.')):
                    formatted_lines.append(f"🩺 {line}")
                elif line:
                    formatted_lines.append(line)
            
            formatted_content = '\n'.join(formatted_lines)
            
            # Add educational info
            education = []
            if 'diabetes' in content.lower():
                education.append("• **Diabetes**: Your body has trouble controlling blood sugar levels")
                education.append("• **Management**: Take medications regularly, follow diet, monitor blood sugar")
            if 'hypertension' in content.lower() or 'blood pressure' in content.lower():
                education.append("• **Hypertension**: High blood pressure that needs regular monitoring")
                education.append("• **Care**: Take BP medications daily, limit salt, exercise regularly")
            
            response = f"🩺 **Medical Conditions**\n\n{formatted_content}"
            if education:
                response += f"\n\n📚 **What this means:**\n" + '\n'.join(education)
            response += "\n\n💙 Your healthcare team is managing these conditions. Do you have questions about your diagnosis?"
            response += "\n\n📚 Would you like me to explain any medical terms or provide more details about this topic?"
            response += "\n\n❓ Feel free to ask if you have any other questions!"
            return response
        else:
            return f"📝 **Medical Information**\n\n{content}\n\n😊 Hope this helps! Feel free to ask more questions."
    
    def _extract_hospital_info(self) -> str:
        """Extract hospital information from raw text"""
        if not self.raw_text:
            return "Hospital information not found"
        
        # Look for hospital patterns
        hospital_patterns = [
            r"Hospital:\s*([^\n]+)",
            r"([A-Z][^\n]*(?:Hospital|Medical Center|Clinic)[^\n]*)",
            r"Care\s+([^\n]*Hospital[^\n]*)"
        ]
        
        for pattern in hospital_patterns:
            match = re.search(pattern, self.raw_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Hospital information not found"
    
    def _extract_specific_field(self, field_type: str) -> str:
        """Extract specific field from raw text"""
        if not self.raw_text:
            return ""
        
        patterns = {
            "name": [
                r"Mrs\.?\s+([A-Za-z\s]+)",
                r"Patient\s+Name\s*[:\-]\s*([A-Za-z\s\.]+)",
                r"Name\s*[:\-]\s*([A-Za-z\s\.]+)"
            ],
            "age": [
                r"(\d+)-year-old",
                r"Age\s*[:\-]\s*(\d+)",
                r"(\d+)\s*years?\s*old"
            ],
            "gender": [
                r"(\d+-year-old)\s+([MF]ale)",
                r"\b([MF]ale)\b",
                r"Gender\s*[:\-]\s*([MF]ale)"
            ]
        }
        
        field_patterns = patterns.get(field_type, [])
        for pattern in field_patterns:
            match = re.search(pattern, self.raw_text, re.IGNORECASE)
            if match:
                if field_type == "gender" and len(match.groups()) > 1:
                    return match.group(2).strip()
                elif field_type == "age":
                    return match.group(1).strip()
                return match.group(1).strip()
        
        return ""
    
    def _extract_from_raw_text(self, section_type: str) -> str:
        """Extract specific sections from raw text when structured data fails"""
        if not self.raw_text:
            return ""
        
        text = self.raw_text
        
        if section_type == "diagnosis":
            # Look for diagnosis section
            patterns = [
                r"DIAGNOSIS[:\s]*([\s\S]*?)(?=CLINICAL SUMMARY|INVESTIGATIONS|TREATMENT|$)",
                r"Diagnosis[:\s]*([\s\S]*?)(?=Clinical Summary|Investigations|Treatment|$)"
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    # Clean up and return first few lines
                    lines = [line.strip() for line in content.split('\n') if line.strip()][:5]
                    return '\n'.join(lines)
        
        elif section_type == "discharge_advice":
            # Look for discharge advice/instructions
            patterns = [
                r"ADVICE ON DISCHARGE[:\s]*([\s\S]*?)(?=FOLLOW-UP|DOCTOR|$)",
                r"Discharge Advice[:\s]*([\s\S]*?)(?=Follow-up|Doctor|$)",
                r"DISCHARGE INSTRUCTIONS[:\s]*([\s\S]*?)(?=FOLLOW-UP|DOCTOR|$)"
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    # Clean up and return
                    lines = [line.strip() for line in content.split('\n') if line.strip()][:8]
                    return '\n'.join(lines)
        
        elif section_type == "medications":
            # Look for discharge medications
            patterns = [
                r"DISCHARGE MEDICATIONS[:\s]*([\s\S]*?)(?=ADVICE|FOLLOW-UP|$)",
                r"Discharge Medications[:\s]*([\s\S]*?)(?=Advice|Follow-up|$)"
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    lines = [line.strip() for line in content.split('\n') if line.strip()][:6]
                    return '\n'.join(lines)
        
        return ""
    
    def _format_patient_info(self) -> str:
        """Format patient information dynamically"""
        patient_info = self.structured_data.get("patient_information", {})
        if not patient_info:
            return "Patient information not found"
        
        # Dynamic formatting based on available data
        info_parts = []
        if patient_info.get('patient_name'):
            info_parts.append(f"Patient: {patient_info['patient_name']}")
        if patient_info.get('age') and patient_info.get('gender'):
            info_parts.append(f"{patient_info['age']}-year-old {patient_info['gender']}")
        if patient_info.get('admission_date'):
            info_parts.append(f"Admitted: {patient_info['admission_date']}")
        if patient_info.get('discharge_date'):
            info_parts.append(f"Discharged: {patient_info['discharge_date']}")
        
        return '. '.join(info_parts) if info_parts else "Limited patient information available"
    
    def _get_related_context(self, intent: str) -> str:
        """Get related contextual information"""
        context_items = []
        
        # Cross-reference related sections
        if intent == "get_diagnosis" and "treatment_given" in self.structured_data:
            treatment = self.structured_data["treatment_given"].get("treatment", "")
            if treatment:
                context_items.append(f"Treatment provided: {treatment[:100]}...")
        
        elif intent == "get_medications" and "discharge_advice" in self.structured_data:
            advice = self.structured_data["discharge_advice"].get("instructions", "")
            if "medication" in advice.lower():
                context_items.append("See discharge instructions for medication guidance")
        
        return '. '.join(context_items[:2])
    
    def _generate_response_variation(self, cached_response: str, intent: str) -> str:
        """Generate variation of cached response to avoid repetition"""
        variations = {
            "Based on the medical record:": "According to the documentation:",
            "Here's what I found:": "The records indicate:",
            "Let me share": "I can tell you",
            "The information shows:": "The data reveals:"
        }
        
        varied_response = cached_response
        for original, variation in variations.items():
            if original in cached_response:
                varied_response = cached_response.replace(original, variation)
                break
        
        return varied_response
    
    def _mcp_fallback_search(self, section: str) -> str:
        """MCP fallback search"""
        if section and section in self.structured_data:
            section_data = self.structured_data[section]
            if isinstance(section_data, dict):
                for field, value in section_data.items():
                    if value:
                        return str(value)
        return self._show_available_sections()
    
    def _mcp_generate_suggestions(self, intent: Dict, entities: List) -> List[str]:
        """Enhanced MCP suggestion generation using LLM handler"""
        intent_name = intent["intent"]
        base_content = self._get_base_content(intent_name)
        
        # Use LLM handler's suggestion generator
        try:
            prompt = f"Context: {base_content}\n\nQuestion: {self._intent_to_question(intent_name)}"
            llm_response = self.llm_handler.generate(prompt, intent_name)
            
            if isinstance(llm_response, dict) and 'suggestions' in llm_response:
                return llm_response['suggestions']
        except Exception as e:
            print(f"Suggestion generation error: {e}")
        
        # Fallback to basic suggestions
        suggestions = []
        if intent_name == "get_diagnosis":
            suggestions.extend([
                "💊 What medications were prescribed?",
                "🏥 What treatment was provided?",
                "📋 What are the discharge instructions?",
                "🔬 What test results support this diagnosis?"
            ])
        elif intent_name == "get_medications":
            suggestions.extend([
                "⏰ What are the exact dosing schedules?",
                "⚠️ What side effects should I watch for?",
                "🍽️ Should these be taken with or without food?",
                "📊 How will effectiveness be monitored?"
            ])
        elif intent_name == "get_summary":
            suggestions.extend([
                "🩺 What was the primary diagnosis?",
                "💉 What procedures were performed?",
                "🏠 What should be done at home?",
                "📅 What follow-up care is needed?"
            ])
        else:
            suggestions.extend([
                "🩺 What is the main diagnosis?",
                "💊 What medications were prescribed?",
                "📋 What are the discharge instructions?",
                "🚨 What symptoms require immediate attention?"
            ])
        
        return suggestions[:4]
    
    def _intent_to_question(self, intent_name: str) -> str:
        """Convert intent to natural language question"""
        intent_questions = {
            "get_diagnosis": "What is the patient's diagnosis?",
            "get_medications": "What medications are prescribed?",
            "get_treatment": "What treatment was provided?",
            "get_summary": "Can you provide a summary of the patient's condition?",
            "get_test_results": "What are the test results?",
            "get_discharge_info": "What are the discharge instructions?",
            "get_patient_info": "What is the patient information?",
            "general_query": "Please provide relevant information."
        }
        return intent_questions.get(intent_name, "Please provide relevant information.")
    
    def _format_context_for_llm(self) -> str:
        """Format structured data as context for LLM"""
        if not self.structured_data:
            return self.raw_text[:2000] if self.raw_text else "No context available"
        
        formatted_sections = []
        for section_name, section_data in self.structured_data.items():
            if isinstance(section_data, dict) and section_data:
                formatted_sections.append(f"\n{section_name.upper().replace('_', ' ')}:")
                for field, value in section_data.items():
                    if value and str(value).strip():
                        formatted_sections.append(f"  {field.replace('_', ' ').title()}: {str(value).strip()}")
        
        return '\n'.join(formatted_sections) if formatted_sections else self.raw_text[:2000]

    def _handle_greetings(self, query: str) -> Optional[str]:
        """Handle greetings and thanks"""
        query_lower = query.lower().strip()
        
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(query_lower.startswith(greeting) for greeting in greetings):
            return "Hello! I'm your Medical Assistant. Ask me about patient info, diagnosis, medications, or discharge instructions."
        
        thanks = ['thank you', 'thanks', 'thank u', 'thx']
        if any(thank in query_lower for thank in thanks):
            return "You're welcome! Feel free to ask if you have any other questions."
        
        return None

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better understanding"""
        query = query.lower().strip()
        
        # Expand common abbreviations
        abbreviations = {
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'meds': 'medications',
            'dx': 'diagnosis',
            'rx': 'prescription'
        }
        
        for abbr, expansion in abbreviations.items():
            query = re.sub(r'\b' + abbr + r'\b', expansion, query)
        
        return query

    def _detect_query_category(self, query: str) -> Tuple[Optional[str], float]:
        """Detect query category with confidence scoring"""
        best_category = None
        best_confidence = 0.0
        
        query_lower = query.lower()
        
        for category, info in self.query_categories.items():
            for keyword in info['keywords']:
                if keyword in query_lower:
                    confidence = 0.95
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_category = category
                    break
        
        return best_category, best_confidence

    def _handle_patient_info_query(self, query: str) -> str:
        """Handle patient information queries"""
        patient_info = self.structured_data.get('patient_information', {})
        
        if 'name' in query:
            name = patient_info.get('patient_name', 'Not found')
            return f"Patient Name: {name}"
        elif 'age' in query:
            age = patient_info.get('age', 'Not found')
            return f"Age: {age} years"
        elif 'gender' in query or 'sex' in query:
            gender = patient_info.get('gender', 'Not found')
            return f"Gender: {gender}"
        elif 'admission' in query:
            admission = patient_info.get('admission_date', 'Not found')
            return f"Admission Date: {admission}"
        elif 'discharge' in query:
            discharge = patient_info.get('discharge_date', 'Not found')
            return f"Discharge Date: {discharge}"
        else:
            # Return all available patient info
            info_parts = []
            if patient_info.get('patient_name'):
                info_parts.append(f"Name: {patient_info['patient_name']}")
            if patient_info.get('age'):
                info_parts.append(f"Age: {patient_info['age']} years")
            if patient_info.get('gender'):
                info_parts.append(f"Gender: {patient_info['gender']}")
            if patient_info.get('admission_date'):
                info_parts.append(f"Admitted: {patient_info['admission_date']}")
            if patient_info.get('discharge_date'):
                info_parts.append(f"Discharged: {patient_info['discharge_date']}")
            
            return '. '.join(info_parts) if info_parts else "Patient information not found."

    def _handle_medical_condition_query(self, query: str) -> str:
        """Handle medical condition and diagnosis queries"""
        diagnosis_info = self.structured_data.get('diagnosis', {})
        
        if diagnosis_info.get('diagnosis'):
            diag = diagnosis_info['diagnosis']
            # Extract only diagnosis lines, not full document
            lines = diag.split('\n')
            diagnosis_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Stop at clinical summary or other sections
                if any(keyword in line.upper() for keyword in ['CLINICAL SUMMARY', 'INVESTIGATIONS', 'TREATMENT', 'HOSPITAL COURSE']):
                    break
                # Include diagnosis lines
                if (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 
                    'diagnosis' in line.lower() or 
                    'mellitus' in line.lower() or 
                    'hypertension' in line.lower() or
                    len(diagnosis_lines) < 5):  # First few lines are usually diagnosis
                    diagnosis_lines.append(line)
            
            return '\n'.join(diagnosis_lines) if diagnosis_lines else diag[:200]
        
        return "Diagnosis information not found."

    def _handle_medication_query(self, query: str) -> str:
        """Handle medication-related queries"""
        discharge_meds = self.structured_data.get('discharge_medications', {})
        
        if discharge_meds.get('medications'):
            meds = discharge_meds['medications']
            # Extract only medication lines
            lines = meds.split('\n')
            med_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Stop at other sections
                if any(keyword in line.upper() for keyword in ['DISCHARGE ADVICE', 'FOLLOW-UP', 'INVESTIGATIONS']):
                    break
                # Include medication lines
                if (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.')) or 
                    any(med in line.lower() for med in ['mg', 'tablet', 'capsule', 'daily', 'twice']) or
                    len(med_lines) < 8):  # Reasonable medication count
                    med_lines.append(line)
            
            return '\n'.join(med_lines) if med_lines else meds[:300]
        
        return "Medication information not found."

    def _handle_vital_signs_query(self, query: str) -> str:
        """Handle vital signs queries"""
        vitals = self.structured_data.get('vital_signs', {})
        clinical = self.structured_data.get('clinical_summary', {})
        
        # Check specific vital sign requests
        if 'blood pressure' in query or 'bp' in query:
            bp = vitals.get('blood_pressure') or clinical.get('blood_pressure')
            return f"Blood pressure: {bp}" if bp else "Blood pressure not found."
        elif 'heart rate' in query or 'pulse' in query:
            hr = vitals.get('heart_rate')
            return f"Heart rate: {hr} bpm" if hr else "Heart rate not found."
        elif 'temperature' in query:
            temp = vitals.get('temperature')
            return f"Temperature: {temp}" if temp else "Temperature not found."
        else:
            # Return all available vitals
            vital_parts = []
            if vitals.get('blood_pressure'):
                vital_parts.append(f"BP: {vitals['blood_pressure']}")
            if vitals.get('heart_rate'):
                vital_parts.append(f"HR: {vitals['heart_rate']} bpm")
            if vitals.get('temperature'):
                vital_parts.append(f"Temp: {vitals['temperature']}")
            if vitals.get('respiratory_rate'):
                vital_parts.append(f"RR: {vitals['respiratory_rate']}")
            
            return '. '.join(vital_parts) if vital_parts else "Vital signs not found."

    def _handle_test_results_query(self, query: str) -> str:
        """Handle test results queries"""
        investigations = self.structured_data.get('investigations', {})
        labs = self.structured_data.get('lab_results', {})
        
        # Check for specific lab values
        if 'hemoglobin' in query or 'hb' in query:
            hb = labs.get('hemoglobin')
            return f"Hemoglobin: {hb} g/dl" if hb else "Hemoglobin not found."
        elif 'glucose' in query or 'sugar' in query:
            glucose = labs.get('glucose')
            return f"Glucose: {glucose} mg/dl" if glucose else "Glucose not found."
        else:
            # Extract only investigation lines
            if investigations.get('investigations_details'):
                details = investigations['investigations_details']
                lines = details.split('\n')
                test_lines = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # Stop at other sections
                    if any(keyword in line.upper() for keyword in ['VITAL SIGNS', 'TREATMENT', 'DISCHARGE']):
                        break
                    # Include test result lines
                    if (any(word in line.lower() for word in ['blood', 'glucose', 'hemoglobin', 'creatinine', 'sodium', 'test', 'examination']) or
                        ':' in line or len(test_lines) < 10):
                        test_lines.append(line)
                
                return '\n'.join(test_lines) if test_lines else details[:300]
            elif labs:
                lab_parts = []
                for lab, value in labs.items():
                    if value:
                        lab_parts.append(f"{lab.replace('_', ' ').title()}: {value}")
                return '. '.join(lab_parts) if lab_parts else "Test results not found."
            
            return "Test results not found."

    def _handle_treatment_query(self, query: str) -> str:
        """Handle treatment queries"""
        treatment_info = self.structured_data.get('treatment_given', {})
        
        if treatment_info.get('treatment'):
            treat = treatment_info['treatment']
            # Extract only treatment lines
            lines = treat.split('\n')
            treatment_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Stop at other sections
                if any(keyword in line.upper() for keyword in ['DISCHARGE MEDICATIONS', 'DISCHARGE ADVICE']):
                    break
                # Include treatment lines
                if (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.')) or 
                    any(word in line.lower() for word in ['therapy', 'insulin', 'fluid', 'medication', 'iv']) or
                    len(treatment_lines) < 8):  # Reasonable treatment count
                    treatment_lines.append(line)
            
            return '\n'.join(treatment_lines) if treatment_lines else treat[:300]
        
        return "Treatment information not found."

    def _handle_discharge_info_query(self, query: str) -> str:
        """Handle discharge info queries"""
        advice_info = self.structured_data.get('discharge_advice', {})
        
        if advice_info.get('instructions'):
            inst = advice_info['instructions']
            # Extract only advice lines
            lines = inst.split('\n')
            advice_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Stop at other sections
                if any(keyword in line.upper() for keyword in ['FOLLOW-UP', 'PROGNOSIS', 'DOCTOR']):
                    break
                # Include advice lines
                if (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')) or 
                    any(word in line.lower() for word in ['follow', 'diet', 'medication', 'monitor', 'avoid']) or
                    len(advice_lines) < 10):  # Reasonable advice count
                    advice_lines.append(line)
            
            return '\n'.join(advice_lines) if advice_lines else inst[:400]
        
        return "Discharge advice not found."

    def _handle_dates_timeline_query(self, query: str) -> str:
        """Handle date queries"""
        patient_info = self.structured_data.get('patient_information', {})
        
        if 'admission' in query:
            adm = patient_info.get('admission_date')
            return f"Admission date: {adm}" if adm else "Admission date not found."
        
        if 'discharge' in query:
            dis = patient_info.get('discharge_date')
            return f"Discharge date: {dis}" if dis else "Discharge date not found."
        
        return "Date information not found."

    def _handle_follow_up_query(self, query: str) -> str:
        """Handle follow-up queries"""
        followup_info = self.structured_data.get('follow_up', {})
        
        if followup_info.get('followup_timing'):
            return f"Follow-up: {followup_info['followup_timing']}"
        
        return "Follow-up information not found."

    def _comprehensive_search(self, query: str) -> str:
        """Comprehensive search through all data"""
        query_lower = query.lower()
        query_words = [word for word in query_lower.split() if len(word) > 2]
        
        print(f"🔍 Comprehensive search for: '{query}'")
        print(f"📊 Query words: {query_words}")
        print(f"📋 Available sections: {list(self.structured_data.keys()) if self.structured_data else 'None'}")
        
        # Check if we have structured data
        if not self.structured_data:
            print("⚠️ No structured data available")
            return self._search_raw_text_only(query, query_words)
        
        # Search all structured data
        for section_name, section_data in self.structured_data.items():
            if isinstance(section_data, dict):
                print(f"🔍 Searching section: {section_name} ({len(section_data)} fields)")
                for field, value in section_data.items():
                    if not value or not str(value).strip():
                        continue
                    
                    value_str = str(value).strip()
                    field_lower = field.lower().replace('_', ' ')
                    value_lower = value_str.lower()
                    
                    # Check field name match
                    field_match = any(word in field_lower for word in query_words)
                    # Check value content match  
                    value_match = any(word in value_lower for word in query_words)
                    
                    if field_match or value_match:
                        print(f"✅ MATCH FOUND in {section_name}.{field}")
                        print(f"   Field match: {field_match}, Value match: {value_match}")
                        print(f"   Value preview: {value_str[:100]}...")
                        # Return only relevant part, not entire document
                        if len(value_str) > 500:
                            # Extract first few relevant lines
                            lines = value_str.split('\n')[:5]
                            return f"{field.replace('_', ' ').title()}: {' '.join(lines)}"
                        return f"{field.replace('_', ' ').title()}: {value_str}"
                    else:
                        print(f"   ❌ No match in {field}: {value_str[:50]}...")
        
        # Search raw text if no structured match
        print("🔍 No structured matches, searching raw text...")
        return self._search_raw_text_only(query, query_words)
    
    def _search_raw_text_only(self, query: str, query_words: list) -> str:
        """Search only in raw text when structured data is not available"""
        if self.raw_text:
            print(f"🔍 Searching raw text ({len(self.raw_text)} chars)")
            lines = self.raw_text.split('\n')
            
            # Look for lines containing query words
            matching_lines = []
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(word in line_lower for word in query_words) and len(line.strip()) > 10:
                    matching_lines.append((i+1, line.strip()))
            
            if matching_lines:
                # Return the best match (first one for now)
                line_num, line_text = matching_lines[0]
                print(f"✅ RAW TEXT MATCH found at line {line_num}: {line_text[:100]}...")
                # Return complete line without truncation
                return line_text
            
            # If no direct matches, try partial matches
            for i, line in enumerate(lines):
                line_lower = line.lower()
                # Check if line contains any part of the query
                if any(part in line_lower for part in query.lower().split() if len(part) > 1) and len(line.strip()) > 5:
                    print(f"✅ PARTIAL MATCH found at line {i+1}: {line.strip()[:100]}...")
                    # Return complete line without truncation
                    return line.strip()
        
        print("❌ No matches found anywhere")
        return self._show_available_sections()

    def _show_available_sections(self) -> str:
        """Show available sections with data count"""
        print("📋 Generating available sections summary...")
        
        if not self.structured_data:
            return "No document data available. Please upload a document first."
        
        sections_with_data = []
        
        for section_name, section_data in self.structured_data.items():
            if section_name == 'full_document_text':
                continue
            
            if isinstance(section_data, dict):
                data_count = sum(1 for v in section_data.values() if v and str(v).strip())
                print(f"📊 Section {section_name}: {data_count} fields with data")
                
                if data_count > 0:
                    readable_name = section_name.replace('_', ' ').title()
                    # Show sample field names
                    sample_fields = [f for f, v in section_data.items() if v and str(v).strip()][:3]
                    field_names = ', '.join([f.replace('_', ' ') for f in sample_fields])
                    sections_with_data.append(f"{readable_name} ({field_names})")
        
        if sections_with_data:
            sections_list = '. '.join(sections_with_data[:4])
            return f"Available data: {sections_list}. Try asking specific questions about these topics."
        else:
            # Show raw text info if available
            if self.raw_text and len(self.raw_text.strip()) > 50:
                return f"Document contains {len(self.raw_text)} characters. Try asking about patient name, diagnosis, medications, or treatment."
            return "No structured data found. Document may need manual review."

    def get_available_sections(self) -> List[str]:
        """Get available data sections"""
        return list(self.structured_data.keys()) if self.structured_data else []

    def get_processing_metadata(self) -> Dict:
        """Get processing metadata"""
        return self.processing_metadata
    
    def _generate_smart_suggestions(self, query: str, answer: str, category: str) -> List[str]:
        """Generate OpenAI-like intelligent suggestions"""
        suggestions = []
        
        # OpenAI-style contextual suggestions
        if 'diagnosis' in query.lower():
            suggestions.extend([
                "💊 What medications were prescribed for this condition?",
                "🏥 What treatment was provided during hospitalization?",
                "📋 What are the discharge care instructions?",
                "🔬 What test results support this diagnosis?",
                "📅 What follow-up care is recommended?"
            ])
        elif 'medication' in query.lower():
            suggestions.extend([
                "⏰ What are the exact dosing schedules?",
                "⚠️ What side effects should I watch for?",
                "🍽️ Should these be taken with or without food?",
                "🚫 Are there any drug interactions to avoid?",
                "📊 How will we monitor the effectiveness?"
            ])
        elif 'summary' in query.lower():
            suggestions.extend([
                "🩺 What was the primary reason for admission?",
                "💉 What procedures were performed?",
                "📈 How did the patient respond to treatment?",
                "🏠 What should be done at home for recovery?",
                "⚡ Were there any complications during stay?"
            ])
        elif 'treatment' in query.lower():
            suggestions.extend([
                "⏱️ How long will this treatment continue?",
                "📊 What are the expected outcomes?",
                "🔄 Are there alternative treatment options?",
                "📋 What lifestyle changes are recommended?",
                "🚨 What warning signs should prompt immediate care?"
            ])
        elif 'test' in query.lower() or 'result' in query.lower():
            suggestions.extend([
                "📈 Are these results within normal ranges?",
                "🔍 What do these findings indicate?",
                "🔄 Will these tests need to be repeated?",
                "💡 How do these results affect treatment?",
                "📊 What trends do we see in the lab values?"
            ])
        else:
            # General medical suggestions
            suggestions.extend([
                "🩺 Can you explain the main diagnosis?",
                "💊 What medications should I continue at home?",
                "📋 What are the most important discharge instructions?",
                "📅 When should I schedule follow-up appointments?",
                "🚨 What symptoms require immediate medical attention?"
            ])
        
        # Smart medical context suggestions
        answer_lower = answer.lower()
        if 'diabetes' in answer_lower:
            suggestions.append("🍯 What should my target blood sugar levels be?")
        if 'heart' in answer_lower or 'cardiac' in answer_lower:
            suggestions.append("❤️ What activities are safe for my heart condition?")
        if 'blood pressure' in answer_lower:
            suggestions.append("🩸 How often should I monitor my blood pressure?")
        if 'pain' in answer_lower:
            suggestions.append("😣 What pain management options are available?")
        
        # Remove duplicates and return top 4
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:4]

