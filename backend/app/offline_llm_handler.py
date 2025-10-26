import re
import string
import random
import hashlib
import json
from typing import Dict, List, Optional, Any
from difflib import SequenceMatcher
from collections import defaultdict
from datetime import datetime
import numpy as np

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available, using rule-based approach")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    print("⚠️ spaCy not available, using basic NLP")

class OfflineLLMHandler:
    def __init__(self):
        # Initialize advanced NLP components
        self._init_nlp_components()
        
        # Medical knowledge base
        self.medical_knowledge = {
            'conditions': {
                'diabetes': ['dm', 'diabetes mellitus', 'diabetic', 'sugar disease'],
                'hypertension': ['htn', 'high blood pressure', 'high bp', 'elevated bp'],
                'heart_disease': ['cardiac', 'heart condition', 'coronary', 'chd']
            },
            'medications': {
                'metformin': ['metformin'],
                'amlodipine': ['amlodipine'],
                'aspirin': ['aspirin']
            }
        }
        
        # Enhanced response generation
        self.response_enhancer = MedicalResponseEnhancer()
        self.suggestion_generator = SmartSuggestionGenerator()
        self.instruction_generator = MedicalInstructionGenerator(self.medical_knowledge)

        # OpenAI-like response templates
        self.response_templates = {
            'medical_professional': self._medical_professional_template,
            'patient_friendly': self._patient_friendly_template,
            'technical_detailed': self._technical_detailed_template,
            'conversational': self._conversational_template,
            'educational': self._educational_template,
            'emergency_aware': self._emergency_aware_template
        }
        
        self.context_awareness = {}
        self.response_history = []
        self.conversation_context = []
        self.personality_traits = {
            'empathy_level': 0.9,
            'technical_depth': 0.8,
            'conversational_tone': 0.95,
            'safety_awareness': 1.0,
            'educational_focus': 0.85
        }
        
        # OpenAI-like capabilities
        self.capabilities = {
            'medical_reasoning': True,
            'contextual_understanding': True,
            'safety_monitoring': True,
            'educational_content': True,
            'personalized_responses': True
        }

        # Medical field patterns with enhanced accuracy
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
            }
        }
        
        # Comprehensive medical terminology
        self.medical_synonyms = {
            "diabetes": ["dm", "diabetes mellitus", "diabetic", "sugar disease", "type 2 diabetes", "t2dm"],
            "hypertension": ["htn", "high blood pressure", "high bp", "elevated bp", "arterial hypertension"],
            "medication": ["medicine", "drug", "prescription", "pills", "tablets", "meds", "pharmaceuticals"],
            "doctor": ["physician", "consultant", "dr", "attending", "medical officer", "clinician"],
            "hospital": ["medical center", "clinic", "healthcare facility", "medical facility", "health center"],
            "heart_disease": ["cardiac condition", "coronary artery disease", "cad", "heart condition", "cardiovascular disease"],
            "blood_pressure": ["bp", "arterial pressure", "systolic", "diastolic"],
            "blood_sugar": ["glucose", "blood glucose", "sugar level", "glycemic level", "hba1c"]
        }
        
        # Emergency keywords for safety monitoring
        self.emergency_keywords = [
            'chest pain', 'difficulty breathing', 'severe headache', 'stroke symptoms',
            'heart attack', 'emergency', 'urgent', 'severe pain', 'bleeding',
            'unconscious', 'seizure', 'allergic reaction'
        ]

    def _init_nlp_components(self):
        """Initialize NLP components if available"""
        self.semantic_model = None
        self.medical_ner = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use lightweight models for better performance
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.medical_ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
                print("✅ Advanced NLP components loaded")
            except Exception as e:
                print(f"⚠️ Could not load advanced NLP: {e}")
    
    def generate(self, prompt: str, intent: str = "general") -> Dict[str, Any]:
        """OpenAI-like response generation with medical intelligence"""
        question = self._extract_question(prompt)
        context = self._extract_context(prompt)
        
        if not context.strip():
            return {
                'answer': "Please upload a medical document first, then I can answer your questions about it.",
                'suggestions': ['Upload a medical document', 'Try asking about patient information', 'Ask about medications or diagnosis'],
                'medical_instructions': [],
                'safety_alerts': [],
                'confidence': 0.0
            }
        
        # Safety monitoring
        safety_alerts = self._check_safety_concerns(question, context)
        
        # Enhanced question processing with medical context
        processed_question = self._preprocess_question(question)
        medical_entities = self._extract_medical_entities(question, context)
        
        # Multi-strategy answer generation
        base_answer = self._multi_strategy_answer(processed_question, context)
        confidence = self._calculate_confidence(base_answer, question, context)
        
        # Apply advanced templating with medical intelligence - force patient_friendly for ChatGPT style
        template_type = 'patient_friendly'  # Always use patient_friendly for consistent ChatGPT-style responses
        template_func = self.response_templates[template_type]
        enhanced_answer = template_func(question, base_answer, intent, medical_entities)
        
        # Generate medical instructions and suggestions
        medical_instructions = self.instruction_generator.generate_instructions(question, base_answer, medical_entities)
        smart_suggestions = self.suggestion_generator.generate_suggestions(question, base_answer, medical_entities, context)
        
        # Apply personality and safety
        final_answer = self._apply_personality_and_safety(enhanced_answer, question, safety_alerts)
        
        # Store conversation context
        self._update_conversation_context(question, final_answer, medical_entities)
        
        return {
            'answer': final_answer,
            'suggestions': smart_suggestions,
            'medical_instructions': medical_instructions,
            'safety_alerts': safety_alerts,
            'confidence': confidence,
            'medical_entities': medical_entities,
            'template_used': template_type
        }
    
    def _check_safety_concerns(self, question: str, context: str) -> List[str]:
        """Monitor for safety concerns and emergency situations"""
        alerts = []
        question_lower = question.lower()
        context_lower = context.lower()
        
        for keyword in self.emergency_keywords:
            if keyword in question_lower or keyword in context_lower:
                alerts.append(f"⚠️ If experiencing {keyword}, seek immediate medical attention or call emergency services")
        
        # Check for medication concerns
        if any(word in question_lower for word in ['side effect', 'adverse reaction', 'allergic']):
            alerts.append("⚠️ For medication concerns or adverse reactions, contact your healthcare provider immediately")
        
        return alerts[:2]  # Limit to 2 most important alerts
    
    def _extract_medical_entities(self, question: str, context: str) -> List[Dict[str, Any]]:
        """Extract medical entities using NLP or rule-based approach"""
        entities = []
        
        if self.medical_ner and TRANSFORMERS_AVAILABLE:
            try:
                # Use transformer-based NER
                ner_results = self.medical_ner(question + " " + context[:500])
                for entity in ner_results:
                    if entity['score'] > 0.8:
                        entities.append({
                            'text': entity['word'],
                            'label': entity['entity'],
                            'confidence': entity['score']
                        })
            except Exception:
                pass
        
        # Rule-based medical entity extraction
        entities.extend(self._rule_based_entity_extraction(question, context))
        
        return entities
    
    def _rule_based_entity_extraction(self, question: str, context: str) -> List[Dict[str, Any]]:
        """Rule-based medical entity extraction"""
        entities = []
        text = (question + " " + context).lower()
        
        # Extract conditions
        for condition, synonyms in self.medical_knowledge['conditions'].items():
            for synonym in [condition] + synonyms:
                if synonym in text:
                    entities.append({
                        'text': condition,
                        'label': 'CONDITION',
                        'confidence': 0.9
                    })
                    break
        
        # Extract medications
        for medication, synonyms in self.medical_knowledge['medications'].items():
            for synonym in synonyms:
                if synonym in text:
                    entities.append({
                        'text': medication,
                        'label': 'MEDICATION',
                        'confidence': 0.85
                    })
                    break
        
        return entities
    
    def _calculate_confidence(self, answer: str, question: str, context: str) -> float:
        """Calculate confidence score for the answer"""
        if not answer or "not found" in answer.lower():
            return 0.2
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on answer quality
        if len(answer) > 50:
            confidence += 0.2
        if ':' in answer:  # Structured answer
            confidence += 0.1
        if any(word in answer.lower() for word in ['mg', 'tablet', 'daily', 'twice']):
            confidence += 0.1  # Medical specificity
        
        # Decrease confidence for vague answers
        if any(phrase in answer.lower() for phrase in ['might be', 'possibly', 'unclear']):
            confidence -= 0.2
        
        return min(max(confidence, 0.0), 1.0)
    
    def _update_conversation_context(self, question: str, answer: str, entities: List[Dict]):
        """Update conversation context for better follow-up responses"""
        self.conversation_context.append({
            'question': question,
            'answer': answer,
            'entities': entities,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 5 interactions
        if len(self.conversation_context) > 5:
            self.conversation_context = self.conversation_context[-5:]
    
    def _medical_professional_template(self, query: str, content: str, intent: str, entities: List[Dict] = None) -> str:
        """Medical professional response template with enhanced clinical context"""
        if not content.strip() or "not found" in content.lower():
            return "The requested clinical data is not documented in the available medical record. Consider reviewing additional documentation or contacting the primary care team."
        
        professional_phrases = [
            "Clinical documentation indicates:",
            "Patient presentation demonstrates:", 
            "Medical assessment reveals:",
            "Documentation reflects:",
            "Clinical findings show:"
        ]
        
        intro = random.choice(professional_phrases)
        
        # Add clinical context based on entities
        clinical_context = self._generate_clinical_context(entities) if entities else ""
        
        if intent == "get_diagnosis":
            response = f"{intro}\n\n**Primary Diagnoses:**\n{content}"
            if clinical_context:
                response += f"\n\n**Clinical Considerations:**\n{clinical_context}"
            response += "\n\n*Recommend correlation with clinical presentation and diagnostic workup.*"
            return response
        elif intent == "get_medications":
            response = f"{intro}\n\n**Discharge Pharmacotherapy:**\n{content}"
            if clinical_context:
                response += f"\n\n**Medication Guidance:**\n{clinical_context}"
            response += "\n\n*Ensure patient counseling on medication adherence and monitoring.*"
            return response
        else:
            response = f"{intro}\n\n{content}"
            if clinical_context:
                response += f"\n\n**Additional Context:**\n{clinical_context}"
            response += "\n\n*Clinical correlation advised.*"
            return response
    
    def _generate_clinical_context(self, entities: List[Dict]) -> str:
        """Generate clinical context based on extracted entities"""
        context_parts = []
        
        for entity in entities:
            if entity['label'] == 'CONDITION':
                condition_info = self.medical_knowledge['conditions'].get(entity['text'])
                if condition_info:
                    context_parts.append(f"• {entity['text'].title()}: {condition_info.get('instructions', '')}")
            elif entity['label'] == 'MEDICATION':
                med_info = self.medical_knowledge['medications'].get(entity['text'])
                if med_info:
                    context_parts.append(f"• {entity['text'].title()}: {med_info.get('purpose', '')}")
        
        return '\n'.join(context_parts[:3])  # Limit to 3 most relevant
    
    def _patient_friendly_template(self, query: str, content: str, intent: str, entities: List[Dict] = None) -> str:
        """ChatGPT-like patient-friendly response template"""
        if not content.strip() or "not found" in content.lower():
            return "😊 I don't see that specific information in your medical records. Let me know what else you'd like to know - I'm here to help explain your health information!"
        
        # Format content with emojis like ChatGPT
        formatted_content = self._format_with_medical_emojis(content, intent)
        
        if intent == "get_patient_info":
            return f"👩‍⚕️ **Patient Information**\n\n{formatted_content}\n\n📝 Is there anything specific about the patient details you'd like me to explain?"
        
        elif intent == "get_diagnosis":
            educational_info = self._get_condition_education(content)
            response = f"🩺 **Medical Conditions**\n\n{formatted_content}"
            if educational_info:
                response += f"\n\n📚 **What this means:**\n{educational_info}"
            response += "\n\n💙 Your healthcare team is managing these conditions. Do you have questions about your diagnosis?"
            return response
        
        elif intent == "get_medications":
            med_guidance = self._get_medication_guidance(content)
            response = f"💊 **Your Medications**\n\n{formatted_content}"
            if med_guidance:
                response += f"\n\n📝 **Important Notes:**\n{med_guidance}"
            response += "\n\n✅ Take these exactly as prescribed. Contact your doctor if you have concerns!"
            return response
        
        elif intent == "get_treatment":
            return f"🏥 **Treatment Provided**\n\n{formatted_content}\n\n😊 This shows the care you received during your hospital stay."
        
        elif intent == "get_test_results":
            return f"🔬 **Test Results**\n\n{formatted_content}\n\n📊 These results help your doctor monitor your health condition."
        
        elif intent == "get_hospital_info":
            return f"🏥 **Hospital Information**\n\n{formatted_content}\n\n📍 This is where the patient received medical care."
        
        elif intent == "get_patient_name":
            return f"👤 **Patient Name**\n\n{formatted_content}\n\n📝 This is the patient's full name."
        
        elif intent == "get_patient_age":
            return f"🎂 **Patient Age**\n\n{formatted_content} years\n\n📝 This is the patient's current age."
        
        elif intent == "get_patient_gender":
            return f"♀️ **Patient Gender**\n\n{formatted_content}\n\n📝 This is the patient's gender."
        
        else:
            return f"📝 **Medical Information**\n\n{formatted_content}\n\n😊 Hope this helps! Feel free to ask more questions."
    
    def _generate_educational_content(self, entities: List[Dict]) -> str:
        """Generate educational content for patients"""
        content_parts = []
        
        for entity in entities:
            if entity['label'] == 'CONDITION':
                condition_info = self.medical_knowledge['conditions'].get(entity['text'])
                if condition_info:
                    content_parts.append(f"• {entity['text'].title()}: {condition_info.get('instructions', '')}")
            elif entity['label'] == 'MEDICATION':
                med_info = self.medical_knowledge['medications'].get(entity['text'])
                if med_info:
                    content_parts.append(f"• {entity['text'].title()}: {med_info.get('instructions', '')}")
        
        return '\n'.join(content_parts[:3])
    
    def _format_with_medical_emojis(self, content: str, intent: str) -> str:
        """Format content with medical emojis like ChatGPT"""
        lines = content.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Add appropriate emojis based on content
            if intent == "get_medications":
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    formatted_lines.append(f"💊 {line}")
                else:
                    formatted_lines.append(line)
            elif intent == "get_diagnosis":
                if line.startswith(('1.', '2.', '3.')):
                    formatted_lines.append(f"🩺 {line}")
                else:
                    formatted_lines.append(line)
            elif intent == "get_treatment":
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    formatted_lines.append(f"🏥 {line}")
                else:
                    formatted_lines.append(line)
            elif intent == "get_patient_info":
                if 'name' in line.lower() or 'patient' in line.lower():
                    formatted_lines.append(f"👤 {line}")
                elif 'age' in line.lower():
                    formatted_lines.append(f"🎂 {line}")
                elif 'gender' in line.lower() or 'female' in line.lower() or 'male' in line.lower():
                    formatted_lines.append(f"♀️ {line}" if 'female' in line.lower() else f"♂️ {line}")
                elif 'admit' in line.lower():
                    formatted_lines.append(f"🏥 {line}")
                elif 'discharge' in line.lower():
                    formatted_lines.append(f"🏠 {line}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _get_condition_education(self, content: str) -> str:
        """Get educational information about conditions"""
        education = []
        content_lower = content.lower()
        
        if 'diabetes' in content_lower:
            education.append("• **Diabetes**: Your body has trouble controlling blood sugar levels")
            education.append("• **Management**: Take medications regularly, follow diet, monitor blood sugar")
        
        if 'hypertension' in content_lower or 'blood pressure' in content_lower:
            education.append("• **Hypertension**: High blood pressure that needs regular monitoring")
            education.append("• **Care**: Take BP medications daily, limit salt, exercise regularly")
        
        if 'gastroenteritis' in content_lower:
            education.append("• **Gastroenteritis**: Stomach infection causing nausea and loose stools")
            education.append("• **Recovery**: Stay hydrated, eat light foods, avoid outside food")
        
        return '\n'.join(education[:4])
    
    def _get_medication_guidance(self, content: str) -> str:
        """Get medication guidance"""
        guidance = []
        content_lower = content.lower()
        
        if 'metformin' in content_lower:
            guidance.append("• **Metformin**: Take with food to avoid stomach upset")
        
        if 'telmisartan' in content_lower:
            guidance.append("• **Telmisartan**: Blood pressure medicine - take at same time daily")
        
        if 'pantoprazole' in content_lower:
            guidance.append("• **Pantoprazole**: Stomach protection - take before meals")
        
        if 'paracetamol' in content_lower:
            guidance.append("• **Paracetamol**: For fever/pain only when needed")
        
        return '\n'.join(guidance[:4])
    
    def _educational_template(self, query: str, content: str, intent: str, entities: List[Dict] = None) -> str:
        """Educational template focusing on patient understanding"""
        if not content.strip():
            return "Let me help you understand your medical information better. What specific aspect would you like me to explain?"
        
        educational_intro = [
            "Let me break this down for you:",
            "Here's what you need to know:",
            "Let me explain this in simple terms:",
            "Understanding your health information:"
        ]
        
        intro = random.choice(educational_intro)
        
        # Simplify medical terminology
        simplified_content = self._simplify_medical_terms(content)
        
        response = f"{intro}\n\n{simplified_content}"
        
        # Add educational context
        if entities:
            educational_tips = self._generate_educational_tips(entities)
            if educational_tips:
                response += f"\n\n**Key Points to Remember:**\n{educational_tips}"
        
        return response
    
    def _emergency_aware_template(self, query: str, content: str, intent: str, entities: List[Dict] = None) -> str:
        """Emergency-aware template with safety prioritization"""
        # Check for emergency indicators
        emergency_indicators = ['chest pain', 'difficulty breathing', 'severe', 'emergency', 'urgent']
        
        if any(indicator in query.lower() for indicator in emergency_indicators):
            return f"🚨 **IMPORTANT**: If you're experiencing emergency symptoms, please call emergency services immediately or go to the nearest emergency room.\n\nRegarding your question: {content}\n\nFor non-emergency concerns, contact your healthcare provider."
        
        return self._conversational_template(query, content, intent, entities)
    
    def _simplify_medical_terms(self, content: str) -> str:
        """Simplify medical terminology for patient understanding"""
        simplifications = {
            'hypertension': 'high blood pressure',
            'diabetes mellitus': 'diabetes (high blood sugar)',
            'myocardial infarction': 'heart attack',
            'cerebrovascular accident': 'stroke',
            'pharmacotherapy': 'medication treatment',
            'therapeutic': 'treatment',
            'prophylactic': 'preventive'
        }
        
        simplified = content
        for medical_term, simple_term in simplifications.items():
            simplified = re.sub(medical_term, simple_term, simplified, flags=re.IGNORECASE)
        
        return simplified
    
    def _generate_educational_tips(self, entities: List[Dict]) -> str:
        """Generate educational tips based on entities"""
        tips = []
        
        for entity in entities:
            if entity['label'] == 'CONDITION':
                condition_info = self.medical_knowledge['conditions'].get(entity['text'])
                if condition_info and 'instructions' in condition_info:
                    tips.append(f"• {condition_info['instructions']}")
        
        return '\n'.join(tips[:3])
    
    def _technical_detailed_template(self, query: str, content: str, intent: str, entities: List[Dict] = None) -> str:
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
    
    def _conversational_template(self, query: str, content: str, intent: str, entities: List[Dict] = None) -> str:
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
    
    def _apply_personality_and_safety(self, response: str, query: str, safety_alerts: List[str]) -> str:
        """Apply personality traits and safety measures to response"""
        # Prepend safety alerts if any
        if safety_alerts:
            response = '\n'.join(safety_alerts) + '\n\n' + response
        
        # Add empathy for sensitive topics
        if any(word in query.lower() for word in ['pain', 'worried', 'concerned', 'scared', 'afraid']):
            if self.personality_traits['empathy_level'] > 0.8:
                response += "\n\n💙 I understand this might be concerning. Your feelings are completely normal, and your healthcare team is here to support you."
        
        # Add educational offers
        if any(word in query.lower() for word in ['how', 'why', 'what does', 'explain']):
            if self.personality_traits['educational_focus'] > 0.7:
                response += "\n\n📚 Would you like me to explain any medical terms or provide more details about this topic?"
        
        # Add follow-up encouragement
        if self.personality_traits['conversational_tone'] > 0.9:
            response += "\n\n❓ Feel free to ask if you have any other questions!"
        
        return response

    def _select_template_type(self, query: str, intent: str) -> str:
        """Select appropriate response template - prioritize patient-friendly for ChatGPT-style responses"""
        query_lower = query.lower()
        
        # Always use patient_friendly template for better ChatGPT-style formatting
        # This ensures consistent emoji usage and friendly tone
        if any(word in query_lower for word in ['clinical', 'medical professional', 'doctor']):
            return 'medical_professional'
        elif any(word in query_lower for word in ['technical', 'detailed analysis', 'comprehensive']):
            return 'technical_detailed'
        else:
            # Default to patient_friendly for ChatGPT-like responses
            return 'patient_friendly'

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
            "medications": "Medications"
        }
        
        display_name = field_display_names.get(field, field.replace('_', ' ').title())
        
        # Add appropriate units or context
        if field == "age":
            if not "year" in value.lower():
                value += " years"
        
        return f"**{display_name}:** {value}"

    def _clean_extracted_value(self, value: str) -> str:
        """Enhanced value cleaning"""
        if not value:
            return value
        
        # Remove extra whitespace
        value = re.sub(r'\s+', ' ', value.strip())
        
        # Remove trailing punctuation except periods in abbreviations
        value = re.sub(r'[,;:\-]+$', '', value)
        
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

    def extractive_answer(self, query: str, docs: List) -> Dict[str, Any]:
        """Enhanced extractive answer generation with full response structure"""
        if not docs:
            return {
                'answer': "No relevant information found in the document.",
                'suggestions': ['Try uploading a medical document', 'Ask about patient information', 'Check if the document contains the information you need'],
                'medical_instructions': [],
                'safety_alerts': [],
                'confidence': 0.0
            }
        
        # Combine document content
        combined_text = " ".join([doc.page_content for doc in docs])
        
        # Use enhanced generation
        return self.generate(f"Context: {combined_text}\n\nQuestion: {query}")


class MedicalResponseEnhancer:
    """Enhances responses with medical intelligence"""
    
    def __init__(self):
        self.medical_patterns = {
            'dosage': r'(\d+(?:\.\d+)?\s*(?:mg|g|ml|units?))',
            'frequency': r'(once|twice|three times|\d+\s*times?)\s*(?:daily|per day|a day)',
            'duration': r'for\s+(\d+\s*(?:days?|weeks?|months?))',
        }
    
    def enhance_medication_info(self, text: str) -> str:
        """Enhance medication information with structured format"""
        enhanced_lines = []
        
        for line in text.split('\n'):
            if any(word in line.lower() for word in ['tablet', 'mg', 'daily', 'twice']):
                # Extract dosage, frequency, duration
                dosage = re.search(self.medical_patterns['dosage'], line, re.IGNORECASE)
                frequency = re.search(self.medical_patterns['frequency'], line, re.IGNORECASE)
                
                enhanced_line = line
                if dosage and frequency:
                    enhanced_line += f" [{dosage.group(1)} {frequency.group(1)}]"
                
                enhanced_lines.append(enhanced_line)
            else:
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)


class SmartSuggestionGenerator:
    """Generates intelligent follow-up suggestions"""
    
    def generate_suggestions(self, query: str, answer: str, entities: List[Dict], context: str) -> List[str]:
        """Generate contextual suggestions based on query and answer"""
        suggestions = []
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Context-aware suggestions
        if 'diagnosis' in query_lower:
            suggestions.extend([
                "💊 What medications are prescribed for this condition?",
                "🏥 What treatment was provided during the hospital stay?",
                "📋 What are the discharge care instructions?",
                "🔬 What test results support this diagnosis?"
            ])
        elif 'medication' in query_lower:
            suggestions.extend([
                "⏰ What are the exact dosing schedules?",
                "⚠️ What side effects should I watch for?",
                "🍽️ Should these be taken with or without food?",
                "📊 How will the effectiveness be monitored?"
            ])
        elif 'test' in query_lower or 'result' in query_lower:
            suggestions.extend([
                "📈 Are these results within normal ranges?",
                "🔍 What do these findings indicate for my health?",
                "🔄 Will these tests need to be repeated?",
                "💡 How do these results affect my treatment plan?"
            ])
        else:
            # General medical suggestions
            suggestions.extend([
                "🩺 Can you explain the main diagnosis?",
                "💊 What medications should I continue at home?",
                "📋 What are the most important discharge instructions?",
                "📅 When should I schedule follow-up appointments?"
            ])
        
        # Entity-based suggestions
        if entities and isinstance(entities, list):
            for entity in entities:
                if isinstance(entity, dict) and entity.get('label') == 'CONDITION':
                    condition = entity.get('text', '')
                    if condition == 'diabetes':
                        suggestions.append("🍯 What should my target blood sugar levels be?")
                    elif condition == 'hypertension':
                        suggestions.append("🩸 How often should I monitor my blood pressure?")
        
        # Safety-based suggestions
        if any(word in answer_lower for word in ['pain', 'symptoms', 'side effects']):
            suggestions.append("🚨 What symptoms require immediate medical attention?")
        
        # Remove duplicates and return top 4
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:4]


class MedicalInstructionGenerator:
    """Generates medical instructions and care guidance"""
    
    def __init__(self, medical_knowledge: Dict):
        self.medical_knowledge = medical_knowledge
    
    def generate_instructions(self, query: str, answer: str, entities: List[Dict]) -> List[str]:
        """Generate relevant medical instructions"""
        instructions = []
        
        # General medical instructions based on query type
        if 'discharge' in query.lower():
            instructions.extend([
                "📅 Schedule follow-up appointments as recommended",
                "💊 Take all medications exactly as prescribed",
                "🚨 Contact your healthcare provider if symptoms worsen"
            ])
        elif 'medication' in query.lower():
            instructions.extend([
                "💊 Take medications exactly as prescribed",
                "⏰ Take at the same time each day",
                "🚨 Contact doctor if you experience side effects"
            ])
        
        return instructions[:4]  # Limit to 4 most relevant instructions