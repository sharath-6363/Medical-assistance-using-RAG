import os
import re
import hashlib
import json
from typing import Dict, List, Optional, Set
from collections import defaultdict

class OfflineDocumentParser:
    def __init__(self):
        self.extraction_patterns = {
            'semantic': self._semantic_extraction,
            'structured': self._structured_extraction, 
            'contextual': self._contextual_extraction
        }
        self.response_cache = {}
        self.content_fingerprints = set()

    def extract_text(self, file_path: str) -> str:
        """Extract text from different file types"""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return ""
        
        file_ext = os.path.splitext(file_path)[1].lower()
        print(f"🔍 Processing file type: {file_ext}")
        
        try:
            if file_ext == '.txt':
                return self._extract_from_txt(file_path)
            elif file_ext == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_ext in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            else:
                # Try as text file
                return self._extract_from_txt(file_path)
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return ""
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"✅ TXT extracted: {len(text)} characters")
            return text
        except Exception as e:
            print(f"❌ TXT error: {e}")
            return ""
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract from PDF file"""
        try:
            # Try PyMuPDF first
            try:
                import fitz
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                print(f"✅ PDF extracted with PyMuPDF: {len(text)} characters")
                return text
            except ImportError:
                pass
            
            # Try pdfplumber
            try:
                import pdfplumber
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                print(f"✅ PDF extracted with pdfplumber: {len(text)} characters")
                return text
            except ImportError:
                pass
            
            # Try PyPDF2
            try:
                import PyPDF2
                text = ""
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                print(f"✅ PDF extracted with PyPDF2: {len(text)} characters")
                return text
            except ImportError:
                pass
            
            print("❌ No PDF libraries available")
            return ""
            
        except Exception as e:
            print(f"❌ PDF error: {e}")
            return ""
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract from DOCX/DOC file"""
        try:
            # Try python-docx
            try:
                from docx import Document
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                
                # Extract tables
                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            text += cell.text + " "
                        text += "\n"
                
                print(f"✅ DOCX extracted: {len(text)} characters")
                return text
            except ImportError:
                pass
            
            # Try docx2txt
            try:
                import docx2txt
                text = docx2txt.process(file_path)
                print(f"✅ DOCX extracted with docx2txt: {len(text)} characters")
                return text
            except ImportError:
                pass
            
            print("❌ No DOCX libraries available")
            return ""
            
        except Exception as e:
            print(f"❌ DOCX error: {e}")
            return ""

    def parse_document(self, text: str) -> Dict[str, Dict[str, str]]:
        """Advanced multi-pattern document parsing"""
        if not text:
            return {}
        
        # Generate content fingerprint for uniqueness
        content_hash = hashlib.md5(text.encode()).hexdigest()
        if content_hash in self.response_cache:
            print("📋 Using cached extraction")
            return self.response_cache[content_hash]
        
        print(f"🔍 Advanced parsing: {len(text)} characters")
        
        # Multi-pattern extraction with error handling
        sections = {}
        for pattern_name, extractor in self.extraction_patterns.items():
            try:
                pattern_sections = extractor(text)
                if pattern_sections:
                    sections.update(pattern_sections)
            except Exception as e:
                print(f"❌ Pattern {pattern_name} failed: {e}")
                continue
        
        # Enhance with contextual relationships (with error handling)
        try:
            sections = self._enhance_with_context(sections, text)
        except Exception as e:
            print(f"❌ Context enhancement failed: {e}")
        
        # Remove duplicates and empty sections (with error handling)
        try:
            sections = self._deduplicate_sections(sections)
        except Exception as e:
            print(f"❌ Deduplication failed: {e}")
            sections = {k: v for k, v in sections.items() if v}
        
        # Cache result
        self.response_cache[content_hash] = sections
        
        print(f"📊 Multi-pattern extraction: {len(sections)} unique sections")
        return sections
    
    def _semantic_extraction(self, text: str) -> Dict[str, Dict[str, str]]:
        """Semantic pattern extraction"""
        sections = {}
        
        # Semantic medical patterns
        semantic_patterns = {
            'patient_demographics': r'(?:Patient|Name|Age|Gender)[:\s]([^\n]+)',
            'medical_history': r'(?:History|Background|Previous)[:\s]([^\n]+)',
            'chief_complaint': r'(?:Complaint|Presenting|Symptoms)[:\s]([^\n]+)',
            'assessment_plan': r'(?:Assessment|Plan|Impression)[:\s]([^\n]+)'
        }
        
        for section, pattern in semantic_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                sections[section] = {'content': '\n'.join(matches)}
        
        return sections
    
    def _structured_extraction(self, text: str) -> Dict[str, Dict[str, str]]:
        """Structured pattern extraction"""
        sections = {}
        
        # Traditional structured extraction
        sections['patient_information'] = self._extract_all_patient_info(text)
        sections['diagnosis'] = self._extract_all_diagnosis(text)
        sections['clinical_summary'] = self._extract_all_clinical(text)
        sections['investigations'] = self._extract_all_investigations(text)
        sections['treatment_given'] = self._extract_all_treatment(text)
        sections['discharge_medications'] = self._extract_all_medications(text)
        sections['discharge_advice'] = self._extract_all_advice(text)
        sections['follow_up'] = self._extract_all_followup(text)
        
        return sections
    
    def _contextual_extraction(self, text: str) -> Dict[str, Dict[str, str]]:
        """Contextual relationship extraction"""
        sections = {}
        
        try:
            # Extract contextual relationships
            lines = text.split('\n')
            context_map = defaultdict(list)
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                # Medical context patterns
                if any(word in line.lower() for word in ['due to', 'caused by', 'resulting in']):
                    context_map['causal_relationships'].append(line)
                elif any(word in line.lower() for word in ['improved', 'worsened', 'stable']):
                    context_map['clinical_progress'].append(line)
                elif any(word in line.lower() for word in ['monitor', 'follow', 'continue']):
                    context_map['ongoing_care'].append(line)
            
            for context_type, content_list in context_map.items():
                if content_list:
                    sections[context_type] = {'relationships': '\n'.join(content_list)}
        except Exception as e:
            print(f"❌ Contextual extraction error: {e}")
        
        return sections
    
    def _enhance_with_context(self, sections: Dict, text: str) -> Dict:
        """Enhance sections with contextual information"""
        if not sections:
            return sections
            
        enhanced = sections.copy()
        
        try:
            # Add cross-references between sections
            for section_name, section_data in sections.items():
                if isinstance(section_data, dict):
                    for field, value in section_data.items():
                        if value and isinstance(value, str):
                            # Find related content in other sections
                            related_content = self._find_related_content(value, sections, section_name)
                            if related_content:
                                enhanced[section_name][f'{field}_related'] = related_content
        except Exception as e:
            print(f"❌ Context enhancement error: {e}")
            return sections
        
        return enhanced
    
    def _find_related_content(self, content: str, all_sections: Dict, current_section: str) -> str:
        """Find related content across sections"""
        if not content or not isinstance(content, str):
            return ''
            
        try:
            content_words = set(content.lower().split())
            related_items = []
            
            for section_name, section_data in all_sections.items():
                if section_name == current_section:
                    continue
                    
                if isinstance(section_data, dict):
                    for field, value in section_data.items():
                        if value and isinstance(value, str):
                            value_words = set(str(value).lower().split())
                            overlap = len(content_words.intersection(value_words))
                            if overlap > 2:  # Significant overlap
                                related_items.append(f"{section_name}: {str(value)[:100]}")
            
            return '\n'.join(related_items[:3]) if related_items else ''
        except Exception:
            return ''
    
    def _deduplicate_sections(self, sections: Dict) -> Dict:
        """Remove duplicate content across sections"""
        if not sections:
            return sections
            
        try:
            seen_content = set()
            deduplicated = {}
            
            for section_name, section_data in sections.items():
                if isinstance(section_data, dict):
                    clean_section = {}
                    for field, value in section_data.items():
                        if value and isinstance(value, str):
                            content_hash = hashlib.md5(str(value).encode()).hexdigest()
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                clean_section[field] = value
                    
                    if clean_section:
                        deduplicated[section_name] = clean_section
            
            return deduplicated
        except Exception as e:
            print(f"❌ Deduplication error: {e}")
            return {k: v for k, v in sections.items() if v}

    def _extract_all_patient_info(self, text: str) -> Dict[str, str]:
        """Extract ALL patient information"""
        info = {}
        lines = text.split('\n')
        
        # Extract all patient fields
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'patient name' in key or key == 'name':
                    info['patient_name'] = value
                elif 'age' in key:
                    age_match = re.search(r'(\d+)', value)
                    if age_match:
                        info['age'] = age_match.group(1)
                elif 'gender' in key or 'sex' in key:
                    info['gender'] = value
                elif 'mrn' in key or 'medical record' in key or 'hospital number' in key:
                    info['hospital_number'] = value
                elif 'date of birth' in key or 'dob' in key:
                    info['date_of_birth'] = value
                elif 'admission' in key:
                    info['admission_date'] = value
                elif 'discharge' in key and 'date' in key:
                    info['discharge_date'] = value
                elif 'physician' in key or 'doctor' in key:
                    info['attending_physician'] = value
                elif 'unit' in key or 'room' in key:
                    info['unit_room'] = value
        
        # Handle "Age / Gender: 63 / Female" format
        for line in lines:
            if 'Age / Gender:' in line:
                parts = line.split(':', 1)[1].strip().split('/')
                if len(parts) >= 2:
                    info['age'] = parts[0].strip()
                    info['gender'] = parts[1].strip()
        
        return info

    def _extract_all_diagnosis(self, text: str) -> Dict[str, str]:
        """Extract ONLY diagnosis list"""
        diagnosis = {}
        lines = text.split('\n')
        
        # Find diagnosis section
        for i, line in enumerate(lines):
            if any(keyword in line.upper() for keyword in ['PRIMARY DIAGNOSIS', 'DIAGNOSIS', 'DIAGNOSES']):
                # Extract only numbered diagnosis items
                diagnosis_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    # Stop at next major section or clinical summary
                    if (self._is_major_section(next_line) or 
                        'CLINICAL SUMMARY' in next_line.upper()):
                        break
                    # Only include numbered diagnosis lines
                    if (next_line.startswith(('1.', '2.', '3.', '4.', '5.')) or
                        (len(diagnosis_lines) == 0 and any(word in next_line.lower() for word in ['diabetes', 'hypertension', 'mellitus']))):
                        diagnosis_lines.append(next_line)
                    j += 1
                
                if diagnosis_lines:
                    diagnosis['diagnosis'] = '\n'.join(diagnosis_lines)
                    break
        
        return diagnosis

    def _extract_all_clinical(self, text: str) -> Dict[str, str]:
        """Extract ALL clinical information including summary"""
        clinical = {}
        
        # Check if entire text is a summary (no section headers)
        if not any(header in text.upper() for header in ['DIAGNOSIS', 'MEDICATIONS', 'TREATMENT', 'INVESTIGATIONS']):
            # Treat entire text as summary
            clinical['summary'] = text.strip()
            return clinical
        
        lines = text.split('\n')
        
        # Find clinical sections
        clinical_keywords = ['HOSPITAL COURSE', 'CLINICAL SUMMARY', 'CLINICAL PRESENTATION', 'HISTORY', 'SUMMARY']
        
        for i, line in enumerate(lines):
            if any(keyword in line.upper() for keyword in clinical_keywords):
                clinical_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    if self._is_major_section(next_line):
                        break
                    clinical_lines.append(next_line)
                    j += 1
                
                if clinical_lines:
                    clinical['summary'] = '\n'.join(clinical_lines)
                    break
        
        # Extract vital signs from anywhere in text
        vital_patterns = {
            'blood_pressure': r'(?:Blood pressure|BP)\s*[:\-]?\s*(\d+/\d+)',
            'heart_rate': r'(?:Heart rate|HR|Pulse)\s*[:\-]?\s*(\d+)',
            'temperature': r'(?:Temperature|Temp)\s*[:\-]?\s*(\d+\.?\d*)',
            'respiratory_rate': r'(?:Respiratory rate|RR)\s*[:\-]?\s*(\d+)',
            'oxygen_saturation': r'(?:SpO2|O2)\s*[:\-]?\s*(\d+)%?'
        }
        
        for vital, pattern in vital_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                clinical[vital] = match.group(1)
        
        return clinical

    def _extract_all_investigations(self, text: str) -> Dict[str, str]:
        """Extract ONLY investigation results"""
        investigations = {}
        lines = text.split('\n')
        
        # Find investigations section
        for i, line in enumerate(lines):
            if 'INVESTIGATIONS' in line.upper():
                inv_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    if (self._is_major_section(next_line) or 
                        'VITAL SIGNS' in next_line.upper()):
                        break
                    # Only include test result lines with values
                    if ((':' in next_line and any(word in next_line.lower() for word in ['blood', 'glucose', 'hemoglobin', 'creatinine', 'sodium', 'wbc', 'hba1c'])) or
                        next_line.startswith('-')):
                        inv_lines.append(next_line)
                    j += 1
                
                if inv_lines:
                    investigations['investigations_details'] = '\n'.join(inv_lines)
                    break
        
        return investigations

    def _extract_all_treatment(self, text: str) -> Dict[str, str]:
        """Extract ONLY treatment list"""
        treatment = {}
        lines = text.split('\n')
        
        # Find treatment sections
        treatment_keywords = ['TREATMENT PROVIDED', 'TREATMENT GIVEN', 'TREATMENT']
        
        for i, line in enumerate(lines):
            if any(keyword in line.upper() for keyword in treatment_keywords):
                treatment_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    if (self._is_major_section(next_line) or 
                        'DISCHARGE MEDICATIONS' in next_line.upper()):
                        break
                    # Only include numbered treatment lines or relevant treatment
                    if (next_line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')) or
                        any(word in next_line.lower() for word in ['iv', 'insulin', 'therapy', 'medication', 'fluid', 'treatment'])):
                        treatment_lines.append(next_line)
                    j += 1
                
                if treatment_lines:
                    treatment['treatment'] = '\n'.join(treatment_lines)
                    break
        
        return treatment

    def _extract_all_medications(self, text: str) -> Dict[str, str]:
        """Extract ONLY medication list"""
        medications = {}
        lines = text.split('\n')
        
        # Find medication sections
        for i, line in enumerate(lines):
            if any(keyword in line.upper() for keyword in ['DISCHARGE MEDICATIONS', 'MEDICATIONS', 'PRESCRIPTIONS']):
                med_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    if (self._is_major_section(next_line) or 
                        'DISCHARGE ADVICE' in next_line.upper()):
                        break
                    # Only include numbered medication lines or lines with dosage
                    if (next_line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')) or
                        any(word in next_line.lower() for word in ['mg', 'daily', 'twice', 'tablet', 'capsule'])):
                        med_lines.append(next_line)
                    j += 1
                
                if med_lines:
                    medications['medications'] = '\n'.join(med_lines)
                    break
        
        return medications

    def _extract_all_advice(self, text: str) -> Dict[str, str]:
        """Extract ONLY discharge advice list"""
        advice = {}
        lines = text.split('\n')
        
        # Find advice sections
        advice_keywords = ['DISCHARGE ADVICE', 'INSTRUCTIONS', 'ADVICE']
        
        for i, line in enumerate(lines):
            if any(keyword in line.upper() for keyword in advice_keywords):
                advice_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    if (self._is_major_section(next_line) or 
                        'FOLLOW-UP' in next_line.upper()):
                        break
                    # Only include numbered advice lines or relevant advice
                    if (next_line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or
                        any(word in next_line.lower() for word in ['follow', 'diet', 'medication', 'monitor', 'avoid', 'take'])):
                        advice_lines.append(next_line)
                    j += 1
                
                if advice_lines:
                    advice['instructions'] = '\n'.join(advice_lines)
                    break
        
        return advice

    def _extract_all_followup(self, text: str) -> Dict[str, str]:
        """Extract ALL follow-up information"""
        followup = {}
        lines = text.split('\n')
        
        # Find follow-up sections
        for i, line in enumerate(lines):
            if any(keyword in line.upper() for keyword in ['FOLLOW-UP', 'FOLLOW UP', 'FOLLOWUP']):
                followup_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    if self._is_major_section(next_line):
                        break
                    followup_lines.append(next_line)
                    j += 1
                
                if followup_lines:
                    followup['followup_timing'] = '\n'.join(followup_lines)
                    break
        
        return followup

    def _is_major_section(self, line: str) -> bool:
        """Check if line is a major section header"""
        if not line.isupper() or len(line) < 5:
            return False
        
        major_sections = [
            'PRIMARY DIAGNOSIS', 'DIAGNOSIS', 'HOSPITAL COURSE', 'CLINICAL SUMMARY',
            'INVESTIGATIONS', 'TREATMENT PROVIDED', 'TREATMENT GIVEN', 'DISCHARGE MEDICATIONS',
            'FOLLOW-UP INSTRUCTIONS', 'DISCHARGE ADVICE', 'DISCHARGE SUMMARY', 'PHYSICIAN'
        ]
        
        return any(section in line for section in major_sections)