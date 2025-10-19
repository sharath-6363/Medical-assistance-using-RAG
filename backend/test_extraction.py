#!/usr/bin/env python3
"""Test document extraction to verify complete text extraction"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.offline_document_parser import OfflineDocumentParser

def test_extraction(file_path):
    """Test document extraction"""
    print(f"\n{'='*60}")
    print(f"Testing extraction for: {file_path}")
    print(f"{'='*60}\n")
    
    parser = OfflineDocumentParser()
    
    try:
        # Extract text
        print("📄 Extracting text...")
        text = parser.extract_text(file_path)
        
        print(f"\n✅ Extraction successful!")
        print(f"📊 Total characters: {len(text)}")
        print(f"📊 Total lines: {len(text.splitlines())}")
        print(f"📊 Total words: {len(text.split())}")
        
        # Show first 1000 characters
        print(f"\n📝 First 1000 characters:")
        print("-" * 60)
        print(text[:1000])
        print("-" * 60)
        
        # Parse document structure
        print(f"\n🔍 Parsing document structure...")
        structured_data = parser.parse_document(text)
        
        print(f"\n✅ Parsed {len(structured_data)} sections:")
        for section, data in structured_data.items():
            print(f"  • {section}: {len(str(data))} chars")
            if isinstance(data, dict):
                for key in data.keys():
                    print(f"    - {key}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test with uploaded files
    upload_dir = Path(__file__).parent / "uploads"
    
    if not upload_dir.exists():
        print("❌ No uploads directory found")
        sys.exit(1)
    
    files = list(upload_dir.glob("*"))
    
    if not files:
        print("❌ No files found in uploads directory")
        sys.exit(1)
    
    print(f"Found {len(files)} files to test")
    
    for file_path in files:
        if file_path.suffix.lower() in ['.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg']:
            test_extraction(str(file_path))
            print("\n")
