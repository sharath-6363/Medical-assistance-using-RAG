#!/usr/bin/env python3
"""
Setup script for NLP components
Run this after installing requirements.txt
"""

import subprocess
import sys
import os

def install_spacy_model():
    """Install spaCy English model"""
    try:
        print("📦 Installing spaCy English model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to install spaCy model: {e}")
        print("You can install it manually with: python -m spacy download en_core_web_sm")

def verify_installations():
    """Verify that all components are working"""
    print("\n🔍 Verifying installations...")
    
    # Test transformers
    try:
        from transformers import pipeline
        print("✅ Transformers available")
    except ImportError:
        print("❌ Transformers not available")
    
    # Test sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers available")
    except ImportError:
        print("❌ Sentence Transformers not available")
    
    # Test spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy with English model available")
    except (ImportError, OSError):
        print("❌ spaCy or English model not available")
    
    # Test torch
    try:
        import torch
        print(f"✅ PyTorch available (version: {torch.__version__})")
    except ImportError:
        print("❌ PyTorch not available")

def main():
    print("🚀 Setting up NLP components for Patient Discharge Assistant")
    print("=" * 60)
    
    install_spacy_model()
    verify_installations()
    
    print("\n" + "=" * 60)
    print("🎉 Setup complete! Your enhanced medical AI is ready to use.")
    print("\nNote: If any components failed to install, the system will")
    print("automatically fall back to rule-based approaches.")

if __name__ == "__main__":
    main()