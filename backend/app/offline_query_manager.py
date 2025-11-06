import re
from typing import Dict, List, Optional, Any

class OfflineQueryManager:
    def __init__(self, forensic_extractor, llm_handler, rag_pipeline=None):
        self.forensic_extractor = forensic_extractor
        self.llm_handler = llm_handler
        self.rag_pipeline = rag_pipeline
        self.structured_data = {}
        self.raw_text = ""

    def process_document(self, text: str, filename: str = None):
        """Process document and extract structured data"""
        print(f"🔄 Processing document: {len(text)} characters")
        self.raw_text = text
        self.structured_data = self._extract_all_data(text)
        print(f"✅ Extracted {len(self.structured_data)} sections")

    def _extract_all_data(self, text: str) -> Dict[str, Any]:
        """Extract data directly from text - no hardcoded fallbacks"""
        # Always extract from the actual text content
        flattened = self._direct_extraction(text)
        
        # Also try forensic extraction if available
        if hasattr(self.forensic_extractor, 'extracted_data') and self.forensic_extractor.extracted_data:
            forensic_data = self.forensic_extractor.extracted_data
            forensic_extracted = self._convert_forensic_to_legacy(forensic_data)
            
<<<<<<< HEAD
            # Merge forensic data with direct extraction (prefer direct extraction)
            for key, value in forensic_extracted.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subvalue and not flattened.get(subkey):
                            flattened[subkey] = subvalue
=======
            for section, data in self.structured_data.items():
                if isinstance(data, dict):
                    icon = section_icons.get(section, '📄')
                    print(f"\n{icon} {section.upper().replace('_', ' ')}:")
                    for key, value in data.items():
                        if value:
                            
                            lines = str(value).split('\n')
                            if len(lines) > 3:
                                print(f"   ✅ {key.replace('_', ' ').title()}:")
                                for i, line in enumerate(lines[:3]):
                                    print(f"      {i+1}. {line}")
                                if len(lines) > 3:
                                    print(f"      ... and {len(lines)-3} more items")
                            else:
                                print(f"   ✅ {key.replace('_', ' ').title()}: {value}")
>>>>>>> a232cd8c699644b2654aa45a7f3be6872b2930a9
        
        return flattened
    
    def _convert_forensic_to_legacy(self, forensic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert forensic extraction format to legacy format"""
        legacy_data = {}
        
        # Extract patient info from forms
        patient_info = {}
        for form in forensic_data.get('forms', []):
            key = form.get('key', '').lower()
            value = form.get('value', '')
            
            if 'name' in key or 'patient' in key:
                patient_info['patient_name'] = value
            elif 'age' in key:
                patient_info['age'] = value
            elif 'gender' in key or 'sex' in key:
                patient_info['gender'] = value
            elif 'admission' in key:
                patient_info['admission_date'] = value
            elif 'discharge' in key and 'date' in key:
                patient_info['discharge_date'] = value
        
        legacy_data['patient_information'] = patient_info
        
        # Extract diagnosis from blocks
        diagnosis_info = {}
        for block in forensic_data.get('blocks', []):
            if block.get('block_type') == 'heading' and 'diagnosis' in block.get('text', '').lower():
                # Find next paragraph blocks for diagnosis content
                diagnosis_info['diagnosis'] = block.get('text', '')
                break
        
        legacy_data['diagnosis'] = diagnosis_info
        
        # Extract other sections from blocks
        clinical_info = {'summary': forensic_data.get('full_text', '')[:500]}
        legacy_data['clinical_summary'] = clinical_info
        
        return legacy_data
    
    def _direct_extraction(self, text: str) -> Dict[str, Any]:
        """Comprehensive extraction of ALL sections from uploaded document"""
        data = {}
        
        # Extract ALL sections dynamically - get complete text for each section
        sections_patterns = {
            'patient_name': [
                r"Patient Name\s*:?\s*([A-Za-z\s\.R]+)",
                r"Name\s*:?\s*([A-Za-z\s\.R]+)",
                r"Mr\.?\s+([A-Za-z\s\.R]+)",
                r"Mrs\.?\s+([A-Za-z\s\.R]+)"
            ],
            'age': [
                r"Age/Gender\s*:?\s*(\d+)\s*/",
                r"Age\s*:?\s*(\d+)",
                r"(\d+)\s*years?\s*old",
                r"(\d+)-year-old"
            ],
            'gender': [
                r"Age/Gender\s*:?\s*\d+\s*/\s*(Male|Female)",
                r"Gender\s*:?\s*(Male|Female)"
            ],
            'patient_id': [
                r"Patient ID\s*:?\s*([A-Za-z0-9]+)",
                r"Hospital Number\s*:?\s*([A-Za-z0-9]+)"
            ],
            'admission_date': [
                r"Date of Admission\s*:?\s*([^\n]+)",
                r"Admission Date\s*:?\s*([^\n]+)"
            ],
            'discharge_date': [
                r"Date of Discharge\s*:?\s*([^\n]+)",
                r"Discharge Date\s*:?\s*([^\n]+)"
            ],
            'doctor': [
                r"Consultant Doctor\s*:?\s*([^\n]+)",
                r"Doctor\s*:?\s*([^\n,]+)",
                r"Attending Physician\s*:?\s*([^\n]+)"
            ],
            'department': [
                r"Department\s*:?\s*([^\n]+)"
            ],
            'ward_room': [
                r"Ward/Room No\s*:?\s*([^\n]+)",
                r"Room\s*:?\s*([^\n]+)"
            ]
        }
        
        # Extract simple fields
        for field, patterns in sections_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if value and len(value) > 1:
                        data[field] = value
                        break
        
        # Extract complete sections with full text - improved patterns
        complete_sections = {
            'chief_complaints': [
                r"CHIEF COMPLAINTS?\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"Chief Complaint\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'history': [
                r"HISTORY OF PRESENT ILLNESS\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"Present History\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"History\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'past_medical_history': [
                r"PAST MEDICAL HISTORY\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'general_examination': [
                r"GENERAL EXAMINATION\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'systemic_examination': [
                r"SYSTEMIC EXAMINATION\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'temperature': [
                r"Temperature\s*:?\s*([\d\.]+\s*°?[CF]?)"
            ],
            'investigations': [
                r"INVESTIGATIONS?\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'diagnosis': [
                r"(?:PRIMARY\s+)?DIAGNOSIS\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"Diagnosis\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'treatment': [
                r"TREATMENT GIVEN\s*(?:DURING HOSPITAL STAY)?\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"Treatment\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'condition_at_discharge': [
                r"CONDITION AT DISCHARGE\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'medications': [
                r"MEDICATIONS ON DISCHARGE\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"DISCHARGE MEDICATIONS?\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'discharge_advice': [
                r"FOLLOW-UP ADVICE\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"DISCHARGE ADVICE\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"DISCHARGE INSTRUCTIONS\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"Follow-up\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"Instructions\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'discharge_instructions': [
                r"DISCHARGE INSTRUCTIONS\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"POST-DISCHARGE INSTRUCTIONS\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"INSTRUCTIONS FOR PATIENT\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'follow_up_advice': [
                r"FOLLOW-UP ADVICE\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"FOLLOW UP\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'clinical_summary': [
                r"CLINICAL SUMMARY\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)",
                r"Clinical Summary\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ],
            'doctors_remarks': [
                r"DOCTOR'S REMARKS\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)"
            ]
        }
        
        # Extract complete sections with better formatting
        for section, patterns in complete_sections.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    # Clean and format content properly
                    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Normalize line breaks
                    content = re.sub(r'^\s*:?\s*', '', content)  # Remove leading colons/spaces
                    content = re.sub(r'\s+$', '', content)  # Remove trailing spaces
                    # Preserve numbered lists and bullet points
                    content = re.sub(r'\n(\d+\.)', '\n\n\1', content)  # Add space before numbered items
                    content = re.sub(r'\n([•-])', '\n\n\1', content)  # Add space before bullet points
                    if content and len(content) > 3:
                        data[section] = content
                        break
        
        # Extract any additional sections dynamically with better formatting
        additional_sections = re.findall(r'^([A-Z][A-Z\s]{2,})\s*:?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Z\s]{4,}:|\Z)', text, re.MULTILINE)
        for section_name, content in additional_sections:
            section_key = section_name.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
            if section_key not in data and content.strip() and len(content.strip()) > 10:
                # Format additional sections properly
                formatted_content = content.strip()
                formatted_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', formatted_content)
                formatted_content = re.sub(r'\n(\d+\.)', '\n\n\1', formatted_content)
                formatted_content = re.sub(r'\n([•-])', '\n\n\1', formatted_content)
                data[section_key] = formatted_content
        
        return data

    def handle_query(self, query: str) -> Dict:
        """Fast query handling"""
        try:
<<<<<<< HEAD
            query_lower = query.lower().strip()
=======
            print(f"🔍 MCP Processing query: '{query}'")
            
            # MCP Protocol 
            mcp_request = MCPRequest(method="query/analyze", params={"query": query})
            mcp_result = self._handle_mcp_request(mcp_request)
            
            return mcp_result
>>>>>>> a232cd8c699644b2654aa45a7f3be6872b2930a9
            
            # Direct field mapping for speed
            if "name" in query_lower:
                patient_name = self.structured_data.get("patient_name", "")
                if not patient_name:
                    patient_name = "No patient name found in the document"
                return self._format_response(patient_name, "get_patient_name")
            elif "age" in query_lower:
                age = self.structured_data.get("age", "")
                if not age:
                    age = "No age information found in the document"
                return self._format_response(age, "get_patient_age")
            elif "gender" in query_lower or "sex" in query_lower:
                gender = self.structured_data.get("gender", "")
                if not gender:
                    gender = "No gender information found in the document"
                return self._format_response(gender, "get_patient_gender")
            elif "diagnosis" in query_lower:
                diagnosis = self.structured_data.get("diagnosis", "")
                if not diagnosis:
                    diagnosis = "No diagnosis information found in the document"
                return self._format_response(diagnosis, "get_diagnosis")
            elif "medication" in query_lower or "medicine" in query_lower:
                medications = self.structured_data.get("medications", "")
                if not medications:
                    medications = "No medication information found in the document"
                return self._format_response(medications, "get_medications")
            elif "instruction" in query_lower or ("discharge" in query_lower and "advice" in query_lower):
                # Look for discharge instructions/advice
                advice = self.structured_data.get("discharge_advice", "")
                if not advice:
                    # Try alternative keys for instructions
                    for key in ['discharge_instructions', 'follow_up_advice', 'instructions', 'advice']:
                        if self.structured_data.get(key):
                            advice = self.structured_data[key]
                            break
                if not advice:
                    advice = "No discharge instructions found in the document"
                return self._format_response(advice, "get_discharge_info")
            elif "summary" in query_lower:
                summary_content = self.structured_data.get("clinical_summary", "")
                if not summary_content:
                    summary_content = "No clinical summary found in the document"
                return self._format_response(summary_content, "get_summary")
            elif "chief" in query_lower and "complaint" in query_lower:
                complaints = self.structured_data.get("chief_complaints", "")
                if not complaints:
                    complaints = "No chief complaints found in the document"
                return self._format_response(complaints, "get_chief_complaints")
            elif "history" in query_lower:
                history = self.structured_data.get("history", "")
                if not history:
                    history = "No history information found in the document"
                return self._format_response(history, "get_history")
            elif "temperature" in query_lower:
                temp = self.structured_data.get("temperature", "")
                if not temp:
                    temp = "No temperature information found in the document"
                return self._format_response(temp, "get_temperature")
            elif "investigation" in query_lower:
                investigations = self.structured_data.get("investigations", "")
                if not investigations:
                    investigations = "No investigation results found in the document"
                return self._format_response(investigations, "get_investigations")
            elif "treatment" in query_lower:
                treatment = self.structured_data.get("treatment", "")
                if not treatment:
                    treatment = "No treatment information found in the document"
                return self._format_response(treatment, "get_treatment")
            elif "doctor" in query_lower:
                doctor = self.structured_data.get("doctor", "")
                if not doctor:
                    doctor = "No doctor information found in the document"
                return self._format_response(doctor, "get_doctor")
            else:
                # Search through ALL available data for any matches
                search_results = []
                query_words = [word for word in query_lower.split() if len(word) > 2]
                
                for key, value in self.structured_data.items():
                    if value and isinstance(value, str):
                        value_lower = value.lower()
                        # Skip disclaimer sections for medical queries
                        if "instruction" in query_lower and "disclaimer" in key.lower():
                            continue
                        # Check if any query words match in this section
                        matches = sum(1 for word in query_words if word in value_lower)
                        if matches > 0:
                            search_results.append((key, value, matches))
                
                if search_results:
                    # Sort by number of matches and return the best one
                    search_results.sort(key=lambda x: x[2], reverse=True)
                    best_match = search_results[0]
                    section_name = best_match[0].replace('_', ' ').title()
                    content = best_match[1]
                    
                    # Format as proper section response with better formatting
                    formatted_content = self._format_section_content(content)
                    answer = f"🔍 **Found in {section_name}:**\n\n{formatted_content}\n\n💡 This information was found based on your search."
                    
                    return {
                        "answer": answer,
                        "category": "search_result",
                        "confidence": 0.9,
                        "suggestions": self._generate_suggestions("search_result"),
                        "medical_instructions": [],
                        "safety_alerts": [],
                        "entities": [],
                        "extracted_data": self.structured_data,
                        "section_found": section_name,
                        "section_key": best_match[0]
                    }
                
                available_info = []
                for key in self.structured_data.keys():
                    if self.structured_data[key]:
                        available_info.append(key.replace('_', ' '))
                
                if available_info:
                    # Create clickable sections for chat window
                    clickable_sections = []
                    section_buttons = []
                    
                    for i, info in enumerate(available_info[:12]):  # Show first 12
                        section_key = list(self.structured_data.keys())[i] if i < len(self.structured_data) else info.replace(' ', '_').lower()
                        clickable_sections.append(f"• **{info}**")
                        section_buttons.append({"key": section_key, "name": info})
                    
                    more_count = len(available_info) - 12
                    sections_text = "\n".join(clickable_sections)
                    
                    if more_count > 0:
                        response = f"📄 **Available Sections (Click to view):**\n\n{sections_text}\n\n*...and {more_count} more sections*\n\n💡 Click any section above to view its content!"
                    else:
                        response = f"📄 **Available Sections (Click to view):**\n\n{sections_text}\n\n💡 Click any section above to view its content!"
                    
                    formatted_response = self._format_response(response, "section_list")
                    formatted_response["section_buttons"] = section_buttons
                    formatted_response["show_sections"] = True
                    return formatted_response
                else:
                    response = "Please upload a document first, then I can answer questions about it."
                    return self._format_response(response, "general_query")
                
        except Exception as e:
            print(f"❌ Query error: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "category": "error",
                "confidence": 0.0,
                "suggestions": []
            }

    def _format_response(self, content: str, intent: str) -> Dict:
        """Format response with ChatGPT-style formatting"""
        
<<<<<<< HEAD
        if intent == "get_patient_name":
            if "not found" in content.lower():
                answer = f"❌ **Patient Name**\n\n{content}\n\n💡 Try asking about other information available in the document."
            else:
                answer = f"👤 **Patient Name**\n\n{content}\n\n📝 This is the patient's full name.\n\n❓ Feel free to ask if you have any other questions!"
=======
        # MCP semantic analysis
        intent = self._mcp_detect_intent(query)
        entities = self._mcp_extract_entities(query)
        context_relevance = self._mcp_assess_context(query)
        
        # Generateing MCP response
        answer = self._mcp_generate_answer(intent, context_relevance)
        suggestions = self._mcp_generate_suggestions(intent, entities)
        
        
        prompt = f"Context: {self._format_context_for_llm()}\n\nQuestion: {query}"
        
        try:
            llm_response = self.llm_handler.generate(prompt, intent["intent"])
            
            if isinstance(llm_response, dict):
                 
                llm_suggestions = llm_response.get('suggestions', [])
                if not isinstance(llm_suggestions, list):
                    llm_suggestions = suggestions
                
                 
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
        
        # Fallback  
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
        
         
        base_content = self._get_base_content(intent_name)
        
        if not base_content or base_content == "Information not found":
            return self._show_available_sections()
        
         
        prompt = f"Context: {base_content}\n\nQuestion: {self._intent_to_question(intent_name)}"
        
        try:
            llm_response = self.llm_handler.generate(prompt, intent_name)
            
 
            if isinstance(llm_response, dict):
                formatted_answer = llm_response.get('answer', base_content)
                 
                return formatted_answer if formatted_answer != base_content else self._format_raw_content(base_content, intent_name)
            else:
                return str(llm_response)
        except Exception as e:
            print(f"LLM generation error: {e}")
             
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
>>>>>>> a232cd8c699644b2654aa45a7f3be6872b2930a9
            
        elif intent == "get_patient_age":
            if "not found" in content.lower():
                answer = f"❌ **Patient Age**\n\n{content}"
            else:
                age_text = content if "year" in content else f"{content} years"
                answer = f"🎂 **Patient Age**\n\n{age_text}\n\n📝 This is the patient's current age.\n\n❓ Feel free to ask if you have any other questions!"
            
        elif intent == "get_patient_gender":
            if "not found" in content.lower():
                answer = f"❌ **Patient Gender**\n\n{content}"
            else:
                emoji = "♀️" if "female" in content.lower() else "♂️"
                answer = f"{emoji} **Patient Gender**\n\n{content}\n\n📝 This is the patient's gender.\n\n❓ Feel free to ask if you have any other questions!"
            
        elif intent == "get_diagnosis":
            if "not found" in content.lower():
                answer = f"❌ **Diagnosis**\n\n{content}"
            else:
                # Format diagnosis with emojis
                lines = content.split('\n')
                formatted_lines = []
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith(('1.', '2.', '3.')) or any(word in line.lower() for word in ['fever', 'hypertension', 'diabetes'])):
                        formatted_lines.append(f"🩺 {line}")
                    elif line:
                        formatted_lines.append(line)
                
                formatted_content = '\n'.join(formatted_lines) if formatted_lines else f"🩺 {content}"
                answer = f"🩺 **Medical Conditions**\n\n{formatted_content}\n\n📚 This shows the patient's diagnosed conditions.\n\n❓ Feel free to ask if you have any other questions!"
            
        elif intent == "get_medications":
            if "not found" in content.lower():
                answer = f"❌ **Medications**\n\n{content}"
            else:
                # Format medications with emojis
                lines = content.split('\n')
                formatted_lines = []
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith(('1.', '2.', '3.', '4.')) or 'mg' in line.lower()):
                        formatted_lines.append(f"💊 {line}")
                    elif line:
                        formatted_lines.append(line)
                
                formatted_content = '\n'.join(formatted_lines)
                answer = f"💊 **Your Medications**\n\n{formatted_content}\n\n✅ Take these exactly as prescribed. Contact your doctor if you have concerns!\n\n❓ Feel free to ask if you have any other questions!"
        
        elif intent == "get_chief_complaints":
            if "not found" in content.lower():
                answer = f"❌ **Chief Complaints**\n\n{content}"
            else:
                answer = f"🤒 **Chief Complaints**\n\n{content}\n\n📝 These are the main symptoms the patient presented with."
        
        elif intent == "get_history":
            if "not found" in content.lower():
                answer = f"❌ **Medical History**\n\n{content}"
            else:
                answer = f"📋 **Medical History**\n\n{content}\n\n📝 This describes the patient's condition and symptoms."
        
        elif intent == "get_temperature":
            if "not found" in content.lower():
                answer = f"❌ **Temperature**\n\n{content}"
            else:
                answer = f"🌡️ **Temperature**\n\n{content}\n\n📝 This is the patient's body temperature reading."
        
        elif intent == "get_investigations":
            if "not found" in content.lower():
                answer = f"❌ **Investigations**\n\n{content}"
            else:
                answer = f"🔬 **Investigation Results**\n\n{content}\n\n📊 These are the test results and findings."
        
        elif intent == "get_treatment":
            if "not found" in content.lower():
                answer = f"❌ **Treatment**\n\n{content}"
            else:
                answer = f"🏥 **Treatment Given**\n\n{content}\n\n💊 This shows the treatment provided during hospital stay."
        
        elif intent == "get_doctor":
            if "not found" in content.lower():
                answer = f"❌ **Doctor Information**\n\n{content}"
            else:
                answer = f"👨‍⚕️ **Consulting Doctor**\n\n{content}\n\n🩺 This is the doctor who treated the patient."
        
        elif intent == "get_discharge_info":
            if "not found" in content.lower():
                answer = f"❌ **Discharge Instructions**\n\n{content}"
            else:
                answer = f"📋 **Discharge Instructions**\n\n{content}\n\n📝 Please follow these instructions carefully for your recovery."
            
        elif intent == "get_summary":
            if "not found" in content.lower():
                answer = f"❌ **Clinical Summary**\n\n{content}"
            else:
                answer = f"📝 **Clinical Summary**\n\n{content}\n\n📊 This summarizes the patient's condition and treatment."
            
        elif intent == "search_result":
            answer = content  # Content already formatted in search logic
        
        elif intent == "section_list":
            answer = content  # Content already formatted with clickable sections
        
        else:
            answer = f"📝 **Medical Information**\n\n{content}\n\n😊 Hope this helps! Feel free to ask more questions.\n\n❓ Feel free to ask if you have any other questions!"

        # Generate suggestions
        suggestions = self._generate_suggestions(intent)
        
        return {
            "answer": answer,
            "category": intent,
            "confidence": 0.95,
            "suggestions": suggestions,
            "medical_instructions": [],
            "safety_alerts": [],
            "entities": [],
            "extracted_data": self.structured_data
        }

    def _generate_suggestions(self, intent: str) -> List[str]:
        """Generate contextual suggestions"""
        suggestions = []
        # Generate suggestions based on available data
        if self.structured_data.get('diagnosis'):
            suggestions.append("🩺 What is the diagnosis?")
        if self.structured_data.get('medications'):
            suggestions.append("💊 What medications were prescribed?")
        if self.structured_data.get('chief_complaints'):
            suggestions.append("🤒 What were the chief complaints?")
        if self.structured_data.get('history'):
            suggestions.append("📋 What is the medical history?")
        if self.structured_data.get('temperature'):
            suggestions.append("🌡️ What was the temperature?")
        if self.structured_data.get('investigations'):
            suggestions.append("🔬 What were the investigation results?")
        if self.structured_data.get('treatment'):
            suggestions.append("🏥 What treatment was given?")
        if self.structured_data.get('discharge_advice') or self.structured_data.get('discharge_instructions'):
            suggestions.append("📋 What are the discharge instructions?")
        
        # Generate suggestions based on ALL available data in document
        suggestion_map = {
            'patient_name': "👤 What is the patient's name?",
            'age': "🎂 What is the patient's age?",
            'gender': "⚧ What is the patient's gender?",
            'diagnosis': "🩺 What is the diagnosis?",
            'medications': "💊 What medications were prescribed?",
            'chief_complaints': "🤒 What were the chief complaints?",
            'history': "📋 What is the medical history?",
            'past_medical_history': "📋 What is the past medical history?",
            'temperature': "🌡️ What was the temperature?",
            'general_examination': "🔍 What were the examination findings?",
            'systemic_examination': "🔬 What were the systemic examination results?",
            'investigations': "🔬 What were the investigation results?",
            'treatment': "🏥 What treatment was given?",
            'discharge_advice': "📋 What are the discharge instructions?",
            'discharge_instructions': "📋 What are the discharge instructions?",
            'follow_up_advice': "📋 What are the follow-up instructions?",
            'condition_at_discharge': "✅ What was the condition at discharge?",
            'doctors_remarks': "💬 What are the doctor's remarks?",
            'doctor': "👨⚕️ Who is the treating doctor?",
            'admission_date': "📅 When was the patient admitted?",
            'discharge_date': "📅 When was the patient discharged?"
        }
        
        for key, suggestion in suggestion_map.items():
            if self.structured_data.get(key) and len(suggestions) < 6:
                suggestions.append(suggestion)
        
        return suggestions[:6] if suggestions else [
            "📄 Please upload a medical document first",
            "💡 Try asking about patient information",
            "🔍 Check what information is available"
        ]
    
    def _format_section_content(self, content: str) -> str:
        """Format section content for better display"""
        if not content:
            return content
        
        # Clean up content
        formatted = content.strip()
        
        # Add proper spacing for numbered lists
        formatted = re.sub(r'(\d+\.)\s*', r'\n\1 ', formatted)
        
        # Add proper spacing for bullet points
        formatted = re.sub(r'([•-])\s*', r'\n\1 ', formatted)
        
        # Clean up extra newlines
        formatted = re.sub(r'\n\s*\n\s*\n+', '\n\n', formatted)
        formatted = re.sub(r'^\n+', '', formatted)
        
        # Add emojis for common medical terms
        formatted = re.sub(r'(\d+\.)\s*(.*(?:mg|tablet|capsule|daily|twice).*)', r'\1 💊 \2', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'(\d+\.)\s*(.*(?:follow|visit|appointment).*)', r'\1 📅 \2', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'(\d+\.)\s*(.*(?:diet|food|eat).*)', r'\1 🍽️ \2', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'(\d+\.)\s*(.*(?:exercise|walk|activity).*)', r'\1 🚶 \2', formatted, flags=re.IGNORECASE)
        
        return formatted
    
    def get_mcp_response(self, query: str) -> str:
        """Get MCP QueryHandler response"""
        if hasattr(self.forensic_extractor, 'answer_query'):
            return self.forensic_extractor.answer_query(query)
        return "MCP QueryHandler not available"