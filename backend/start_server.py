"""
Simple server startup script
Run: python start_server.py
"""

import uvicorn
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    try:
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=False
        )
    except KeyboardInterrupt:
        print("\n✅ Server stopped")
