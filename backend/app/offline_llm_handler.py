import re
import string
import random
import hashlib
from typing import Dict, List, Optional
from difflib import SequenceMatcher
from collections import defaultdict

class OfflineLLMHandler:
    def __init__(self):
        # Advanced response templates
        self.response_templates = {
            'medical_professional': self._medical_professional_template,
            'patient_friendly': self._patient_friendly_template,
            'technical_detailed': self._technical_detailed_template,
            'conversational': self._conversational_template
        }
        self.context_awareness = {}
        self.response_history = []
        self.personality_traits = {
            'empathy_level': 0.8,
            'technical_depth': 0.7,
            'conversational_tone': 0.9
        }
        # Enhanced medical field patterns with confidence scoring
        self.field_patterns = {
            "patient_name": {
                "keywords": ["name", "patient name", "full name", "pt name", "patient", "mr", "mrs", "ms", "dr"],
                "patterns": [
                    r"(?:Patient\s+)?Name\s*[:\-]\s*([A-Za-z\s\.]+)",
                    r"(?:Mr|Mrs|Ms|Dr)\.?\s+([A-Za-z\s]+)",
                    r"Patient\s*[:\-]\s*([A-Za-z\s\.]+)"
                ],
                "priority": 10,
                "confidence_threshold": 0.7
            },
            "age": {
                "keywords": ["age", "how old", "years old", "y/o", "yo", "years", "yrs"],
                "patterns": [
                    r"Age\s*[:\-]\s*(\d+)\s*(?:years?|yrs?|Y)?",
                    r"\(Age[:\s]*(\d+)[^\)]*\)",
                    r"(\d+)\s*(?:year|yr)\s*old"
                ],
                "priority": 9,
                "confidence_threshold": 0.8
            },
            "gender": {
                "keywords": ["gender", "sex", "male", "female", "m", "f"],
                "patterns": [
                    r"Gender\s*[:\-]\s*([MF]ale|[MF])",
                    r"Sex\s*[:\-]\s*([MF]ale|[MF])",
                    r"\b([MF]ale)\b"
                ],
                "priority": 8,
                "confidence_threshold": 0.8
            },
            "diagnosis": {
                "keywords": ["diagnosis", "condition", "problem", "illness", "disease", "diagnosed with"],
                "patterns": [
                    r"(?:Discharge\s+)?Diagnosis\s*[:\-]\s*([^\n]+)",
                    r"Diagnosed\s+with\s*[:\-]?\s*([^\n]+)",
                    r"Primary\s+Diagnosis\s*[:\-]\s*([^\n]+)"
                ],
                "priority": 9,
                "confidence_threshold": 0.7
            },
            "medications": {
                "keywords": ["medication", "medicine", "drug", "prescription", "pills", "tablets", "meds"],
                "patterns": [
                    r"(?:Discharge\s+)?Medications?\s*[:\-]\s*([^\n]+(?:\n[^\n]*)*?)(?=\n\s*[A-Z]|$)",
                    r"Prescribed\s*[:\-]\s*([^\n]+)",
                    r"Take\s+([^\n]+)"
                ],
                "priority": 9,
                "confidence_threshold": 0.6
            },
            "blood_pressure": {
                "keywords": ["blood pressure", "bp", "pressure", "systolic", "diastolic"],
                "patterns": [
                    r"(?:BP|Blood\s+Pressure)\s*[:\-]?\s*(\d+/\d+)",
                    r"(\d+/\d+)\s*mmHg",
                    r"Systolic\s*[:\-]?\s*(\d+)"
                ],
                "priority": 7,
                "confidence_threshold": 0.8
            },
            "blood_sugar": {
                "keywords": ["blood sugar", "glucose", "sugar level", "bs", "fbs", "ppbs"],
                "patterns": [
                    r"(?:Blood\s+Sugar|Glucose|FBS|PPBS)\s*[:\-]?\s*(\d+\.?\d*)\s*(?:mg/dl)?",
                    r"Sugar\s+Level\s*[:\-]?\s*(\d+\.?\d*)",
                    r"(\d+\.?\d*)\s*mg/dl"
                ],
                "priority": 8,
                "confidence_threshold": 0.7
            },
            "admission_date": {
                "keywords": ["admission", "admitted", "admission date", "when admitted"],
                "patterns": [
                    r"(?:Date\s+of\s+)?Admission\s*[:\-]\s*([0-9\-/]+)",
                    r"Admitted\s+on\s*[:\-]?\s*([0-9\-/]+)",
                    r"Admission\s+Date\s*[:\-]\s*([0-9\-/]+)"
                ],
                "priority": 8,
                "confidence_threshold": 0.8
            },
            "discharge_date": {
                "keywords": ["discharge", "discharged", "discharge date", "when discharged"],
                "patterns": [
                    r"(?:Date\s+of\s+)?Discharge\s*[:\-]\s*([0-9\-/]+)",
                    r"Discharged\s+on\s*[:\-]?\s*([0-9\-/]+)",
                    r"Discharge\s+Date\s*[:\-]\s*([0-9\-/]+)"
                ],
                "priority": 8,
                "confidence_threshold": 0.8
            },
            "follow_up": {
                "keywords": ["follow up", "follow-up", "next visit", "appointment", "return"],
                "patterns": [
                    r"Follow\s*[-\s]*up\s+(?:after|in)\s+([^.\n]+)",
                    r"Next\s+visit\s+(?:after|in)\s+([^.\n]+)",
                    r"Return\s+(?:after|in)\s+([^.\n]+)"
                ],
                "priority": 7,
                "confidence_threshold": 0.6
            }
        }
        
        # Medical context understanding
        self.medical_synonyms = {
            "diabetes": ["dm", "diabetes mellitus", "diabetic", "sugar disease"],
            "hypertension": ["htn", "high blood pressure", "high bp", "elevated bp"],
            "medication": ["medicine", "drug", "prescription", "pills", "tablets", "meds"],
            "doctor": ["physician", "consultant", "dr", "attending", "medical officer"],
            "hospital": ["medical center", "clinic", "healthcare facility", "medical facility"]
        }

    def generate(self, prompt: str, intent: str = "general") -> str:
        """Advanced response generation with multiple templates"""
        question = self._extract_question(prompt)
        context = self._extract_context(prompt)
        
        if not context.strip():
            return "Please upload a medical document first, then I can answer your questions about it."
        
        # Enhanced question processing
        processed_question = self._preprocess_question(question)
        
        # Multi-strategy answer generation
        base_answer = self._multi_strategy_answer(processed_question, context)
        
        # Apply advanced templating
        template_type = self._select_template_type(question, intent)
        template_func = self.response_templates[template_type]
        enhanced_answer = template_func(question, base_answer, intent)
        
        # Apply personality traits
        final_answer = self._apply_personality(enhanced_answer, question)
        
        # Store in history
        self.response_history.append({
            'query': question,
            'response': final_answer,
            'template': template_type,
            'intent': intent
        })
        
        return final_answer
    
    def _select_template_type(self, query: str, intent: str) -> str:
        """Select appropriate response template"""
        query_lower = query.lower()
        
        # Avoid repetitive templates
        recent_templates = [h['template'] for h in self.response_history[-3:]]
        
        if any(word in query_lower for word in ['explain', 'what does', 'help me understand']):
            return 'patient_friendly' if 'patient_friendly' not in recent_templates else 'conversational'
        elif any(word in query_lower for word in ['clinical', 'medical', 'professional']):
            return 'medical_professional' if 'medical_professional' not in recent_templates else 'technical_detailed'
        elif any(word in query_lower for word in ['details', 'comprehensive', 'complete']):
            return 'technical_detailed' if 'technical_detailed' not in recent_templates else 'medical_professional'
        else:
            return 'conversational'
    
    def _medical_professional_template(self, query: str, content: str, intent: str) -> str:
        """Medical professional response template"""
        if not content.strip() or "not found" in content.lower():
            return "The requested clinical data is not documented in the available medical record."
        
        professional_phrases = [
            "Clinical documentation indicates:",
            "Patient presentation shows:", 
            "Medical assessment reveals:",
            "Documentation reflects:"
        ]
        
        intro = random.choice(professional_phrases)
        
        if intent == "get_diagnosis":
            return f"{intro}\n\n**Primary Diagnoses:**\n{content}\n\nRecommend correlation with clinical presentation and diagnostic workup."
        elif intent == "get_medications":
            return f"{intro}\n\n**Discharge Pharmacotherapy:**\n{content}\n\nEnsure patient counseling on medication adherence and monitoring."
        else:
            return f"{intro}\n\n{content}\n\nClinical correlation advised."
    
    def _patient_friendly_template(self, query: str, content: str, intent: str) -> str:
        """Patient-friendly response template"""
        if not content.strip() or "not found" in content.lower():
            return "I don't see that information in your medical records. Let me know if you'd like me to look for something else!"
        
        friendly_intros = [
            "Let me explain what your medical records show:",
            "Here's what I found in your health information:",
            "Your medical documents tell us:",
            "Based on your records, here's what we know:"
        ]
        
        intro = random.choice(friendly_intros)
        
        if intent == "get_diagnosis":
            return f"{intro}\n\n{content}\n\nDon't worry - your healthcare team is working to manage these conditions effectively. Do you have any questions about what this means?"
        elif intent == "get_medications":
            return f"{intro}\n\n**Your Medications:**\n{content}\n\nRemember to take these exactly as prescribed. If you have concerns about any medication, please contact your healthcare provider."
        else:
            return f"{intro}\n\n{content}\n\nI hope this helps clarify things for you!"
    
    def _technical_detailed_template(self, query: str, content: str, intent: str) -> str:
        """Technical detailed response template"""
        if not content.strip() or "not found" in content.lower():
            return "Comprehensive analysis of available medical documentation does not contain the requested information parameters."
        
        technical_intros = [
            "Detailed medical record analysis reveals:",
            "Comprehensive documentation review shows:",
            "Systematic evaluation of clinical data indicates:",
            "Thorough assessment of medical information demonstrates:"
        ]
        
        intro = random.choice(technical_intros)
        
        # Add technical context
        lines = content.split('\n')
        formatted_content = '\n'.join([f"• {line.strip()}" for line in lines if line.strip()])
        
        return f"{intro}\n\n{formatted_content}\n\n**Clinical Significance:** This information should be interpreted within the broader clinical context and patient presentation."
    
    def _conversational_template(self, query: str, content: str, intent: str) -> str:
        """Conversational response template"""
        if not content.strip() or "not found" in content.lower():
            return "Hmm, I don't see that information in the document. What else would you like to know?"
        
        conversational_starters = [
            "Great question! Here's what I found:",
            "I can help with that. The records show:",
            "Sure thing! Looking at the information:",
            "Absolutely! Here's what the document tells us:"
        ]
        
        starter = random.choice(conversational_starters)
        
        return f"{starter}\n\n{content}\n\nAnything else you'd like to know about this?"
    
    def _apply_personality(self, response: str, query: str) -> str:
        """Apply personality traits to response"""
        # Add empathy for sensitive topics
        if any(word in query.lower() for word in ['pain', 'worried', 'concerned', 'scared']):
            if self.personality_traits['empathy_level'] > 0.7:
                response += "\n\nI understand this might be concerning. Please don't hesitate to discuss any worries with your healthcare team."
        
        # Add technical depth when appropriate
        if "how" in query.lower() or "why" in query.lower():
            if self.personality_traits['technical_depth'] > 0.6:
                response += "\n\nWould you like me to explain any of these terms in more detail?"
        
        return response

    def _preprocess_question(self, question: str) -> str:
        """Preprocess question for better understanding"""
        # Expand medical abbreviations
        for term, synonyms in self.medical_synonyms.items():
            for synonym in synonyms:
                pattern = r'\b' + re.escape(synonym) + r'\b'
                question = re.sub(pattern, term, question, flags=re.IGNORECASE)
        
        # Normalize question format
        question = question.lower().strip()
        question = re.sub(r'\s+', ' ', question)
        
        return question

    def _multi_strategy_answer(self, question: str, context: str) -> str:
        """Multi-strategy answer generation"""
        # Strategy 1: Direct field matching with confidence scoring
        field_answer = self._enhanced_field_matching(question, context)
        if field_answer and field_answer["confidence"] > 0.7:
            return field_answer["answer"]
        
        # Strategy 2: Pattern-based extraction
        pattern_answer = self._pattern_based_extraction(question, context)
        if pattern_answer:
            return pattern_answer
        
        # Strategy 3: Contextual search
        contextual_answer = self._contextual_search(question, context)
        if contextual_answer:
            return contextual_answer
        
        # Fallback: Intelligent suggestions
        return self._intelligent_fallback(question, context)

    def _enhanced_field_matching(self, question: str, context: str) -> Optional[Dict]:
        """Enhanced field matching with confidence scoring"""
        best_match = None
        best_confidence = 0
        
        question_words = set(question.lower().split())
        
        for field, field_info in self.field_patterns.items():
            # Calculate keyword similarity
            keyword_score = self._calculate_keyword_similarity(question_words, field_info["keywords"])
            
            # Combined confidence score
            confidence = keyword_score * (field_info["priority"] / 10)
            
            if confidence > best_confidence and confidence > field_info["confidence_threshold"]:
                # Try to extract value using patterns
                value = self._extract_field_value_enhanced(field, context, field_info["patterns"])
                if value:
                    best_match = {
                        "field": field,
                        "value": value,
                        "confidence": confidence,
                        "answer": self._format_field_answer(field, value)
                    }
                    best_confidence = confidence
        
        return best_match

    def _calculate_keyword_similarity(self, question_words: set, keywords: List[str]) -> float:
        """Calculate keyword similarity score"""
        max_score = 0
        for keyword in keywords:
            keyword_words = set(keyword.lower().split())
            if keyword_words.issubset(question_words):
                score = len(keyword_words) / len(question_words)
                max_score = max(max_score, score)
            else:
                overlap = len(question_words & keyword_words)
                if overlap > 0:
                    score = overlap / max(len(question_words), len(keyword_words))
                    max_score = max(max_score, score)
        return max_score

    def _extract_field_value_enhanced(self, field: str, context: str, patterns: List[str]) -> Optional[str]:
        """Enhanced field value extraction with multiple patterns"""
        for pattern in patterns:
            try:
                matches = re.finditer(pattern, context, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    value = match.group(1).strip()
                    if value and len(value) > 1:
                        # Validate extracted value
                        if self._validate_extracted_value(field, value):
                            return self._clean_extracted_value(value)
            except Exception:
                continue
        return None

    def _validate_extracted_value(self, field: str, value: str) -> bool:
        """Validate extracted value based on field type"""
        if field == "age":
            try:
                age = int(re.search(r'\d+', value).group())
                return 0 < age < 150
            except:
                return False
        
        elif field == "blood_pressure":
            return bool(re.match(r'\d+/\d+', value))
        
        elif field == "blood_sugar":
            try:
                sugar = float(re.search(r'\d+\.?\d*', value).group())
                return 50 < sugar < 800
            except:
                return False
        
        elif field in ["admission_date", "discharge_date"]:
            return bool(re.match(r'\d+[/-]\d+[/-]\d+', value))
        
        elif field == "patient_name":
            return len(value.split()) >= 1 and value.replace(' ', '').replace('.', '').isalpha()
        
        return True

    def _pattern_based_extraction(self, question: str, context: str) -> Optional[str]:
        """Pattern-based extraction for specific question types"""
        question_patterns = {
            "what_is": r"what\s+(?:is|are)\s+(?:the\s+)?(.+)",
            "how_much": r"how\s+(?:much|many)\s+(.+)",
            "when": r"when\s+(?:was|is|did|will)\s+(.+)",
            "where": r"where\s+(?:is|was|will)\s+(.+)",
            "who": r"who\s+(?:is|was|will)\s+(.+)"
        }
        
        for q_type, pattern in question_patterns.items():
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                target = match.group(1).strip()
                return self._extract_by_question_type(q_type, target, context)
        
        return None

    def _extract_by_question_type(self, q_type: str, target: str, context: str) -> Optional[str]:
        """Extract information based on question type"""
        if q_type == "what_is":
            patterns = [
                rf"{re.escape(target)}\s*[:\-]\s*([^.\n]+)",
                rf"({target}[^.\n]*)",
                rf"([^.\n]*{re.escape(target)}[^.\n]*)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    return f"Regarding {target}: {match.group(1).strip()}"
        
        elif q_type == "when":
            date_patterns = [
                rf"{re.escape(target)}[^.\n]*?(\d+[/-]\d+[/-]\d+)",
                rf"(\d+[/-]\d+[/-]\d+)[^.\n]*?{re.escape(target)}"
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    return f"{target.title()} date: {match.group(1)}"
        
        elif q_type == "who":
            name_patterns = [
                rf"{re.escape(target)}[^.\n]*?(?:Dr\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                rf"(?:Dr\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[^.\n]*?{re.escape(target)}"
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    return f"{target.title()}: {match.group(1)}"
        
        return None

    def _contextual_search(self, question: str, context: str) -> Optional[str]:
        """Contextual search using keyword matching"""
        question_lower = question.lower()
        
        # Direct field search in structured format
        field_patterns = [
            r'✅\s*([^:]+):\s*([^\n]+)',
            r'([A-Z][^:]+):\s*([^\n]+)',
            r'•\s*([^:]+):\s*([^\n]+)'
        ]
        
        for pattern in field_patterns:
            matches = re.findall(pattern, context)
            for field, value in matches:
                if any(word in field.lower() or word in value.lower() for word in question_lower.split() if len(word) > 2):
                    return f"**{field.strip()}:** {value.strip()}"
        
        # Fallback to sentence search
        question_words = question_lower.split()
        stop_words = {"what", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in question_words if word not in stop_words and len(word) > 2]
        
        if keywords:
            lines = context.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in keywords) and ':' in line:
                    return line.strip()
        
        return None

    def _intelligent_fallback(self, question: str, context: str) -> str:
        """Intelligent fallback with suggestions"""
        # Analyze available information
        available_info = self._analyze_available_information(context)
        
        suggestions = []
        if "patient" in question.lower():
            if available_info.get("patient_info"):
                suggestions.extend([
                    "Ask about the patient's name, age, or gender",
                    "Ask about admission or discharge dates"
                ])
        
        if any(word in question.lower() for word in ["medication", "medicine", "drug"]):
            if available_info.get("medications"):
                suggestions.append("Ask 'what medications are prescribed?'")
        
        if any(word in question.lower() for word in ["diagnosis", "condition", "problem"]):
            if available_info.get("diagnosis"):
                suggestions.append("Ask 'what is the diagnosis?'")
        
        if suggestions:
            return f"I found information in the document that might help. Try asking:\n• " + "\n• ".join(suggestions[:3])
        else:
            return "I can help you understand the medical document. Try asking about patient information, diagnosis, medications, test results, or discharge instructions."

    def _analyze_available_information(self, context: str) -> Dict[str, bool]:
        """Analyze what information is available in the context"""
        info_indicators = {
            "patient_info": ["name", "age", "gender", "patient"],
            "medications": ["medication", "prescription", "drug", "tablet"],
            "diagnosis": ["diagnosis", "condition", "disease"],
            "tests": ["test", "result", "lab", "investigation"],
            "vitals": ["blood pressure", "heart rate", "temperature"],
            "dates": ["admission", "discharge", "date"]
        }
        
        available = {}
        context_lower = context.lower()
        
        for category, indicators in info_indicators.items():
            available[category] = any(indicator in context_lower for indicator in indicators)
        
        return available

    def _format_field_answer(self, field: str, value: str) -> str:
        """Format field answer for better presentation"""
        field_display_names = {
            "patient_name": "Patient Name",
            "age": "Age",
            "gender": "Gender",
            "diagnosis": "Diagnosis",
            "medications": "Medications",
            "blood_pressure": "Blood Pressure",
            "blood_sugar": "Blood Sugar Level",
            "admission_date": "Admission Date",
            "discharge_date": "Discharge Date",
            "follow_up": "Follow-up Instructions"
        }
        
        display_name = field_display_names.get(field, field.replace('_', ' ').title())
        
        # Add appropriate units or context
        if field == "blood_pressure":
            value += " mmHg"
        elif field == "blood_sugar":
            if not "mg/dl" in value.lower():
                value += " mg/dl"
        elif field == "age":
            if not "year" in value.lower():
                value += " years"
        
        return f"**{display_name}:** {value}"

    def _post_process_answer(self, answer: str) -> str:
        """Post-process answer for better readability"""
        if not answer:
            return "I couldn't find specific information to answer your question in the document."
        
        # Clean up formatting
        answer = re.sub(r'\s+', ' ', answer.strip())
        
        # Ensure proper capitalization
        answer = answer[0].upper() + answer[1:] if answer else answer
        
        # Add period if missing
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        return answer

    def _clean_extracted_value(self, value: str) -> str:
        """Enhanced value cleaning"""
        if not value:
            return value
        
        # Remove extra whitespace
        value = re.sub(r'\s+', ' ', value.strip())
        
        # Remove trailing punctuation except periods in abbreviations
        value = re.sub(r'[,;:\-]+$', '', value)
        
        # Fix common OCR errors
        value = re.sub(r'\s+([.,;:])', r'\1', value)
        value = re.sub(r'([.,;:])\s*([A-Z])', r'\1 \2', value)
        
        # Capitalize first letter
        if value:
            value = value[0].upper() + value[1:]
        
        return value.strip()

    def _extract_question(self, prompt: str) -> str:
        """Extract question from prompt"""
        match = re.search(r"Question:\s*(.*?)(?:\n|$)", prompt, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else prompt.strip()

    def _extract_context(self, prompt: str) -> str:
        """Extract context from prompt"""
        match = re.search(r"Context\s*:?\s*\n(.*?)(?:\n\nQuestion:|$)", prompt, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Alternative pattern
        match = re.search(r"Context\s*\n(.*?)(?:\n\nQuestion:|$)", prompt, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else prompt

    def extractive_answer(self, query: str, docs: List) -> str:
        """Enhanced extractive answer generation"""
        if not docs:
            return "No relevant information found in the document."
        
        # Combine document content
        combined_text = " ".join([doc.page_content for doc in docs])
        
        # Use enhanced extraction
        return self.generate(f"Context: {combined_text}\n\nQuestion: {query}")