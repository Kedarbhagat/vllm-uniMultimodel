#!/usr/bin/env python
"""
Quick launcher for Title Generation Service with automatic dependency check
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required = ["transformers", "torch", "fastapi", "uvicorn"]
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} missing")
            missing.append(package)
    
    if missing:
        print(f"\n📦 Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            *missing, "-q"
        ])
        print("✅ Dependencies installed\n")

def main():
    print("=" * 60)
    print("  Title Generation Service Launcher")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    check_dependencies()
    
    # Get configuration from environment
    port = os.getenv("TITLE_SERVICE_PORT", "8001")
    host = os.getenv("TITLE_SERVICE_HOST", "0.0.0.0")
    
    print(f"🚀 Starting Title Service on {host}:{port}")
    print(f"   Health check: http://localhost:{port}/health")
    print(f"   API docs: http://localhost:{port}/docs")
    print()
    
    # Import and run the service
    try:
        from title_service import app
        import uvicorn
        
        uvicorn.run(
            app,
            host=host,
            port=int(port),
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 Service stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
