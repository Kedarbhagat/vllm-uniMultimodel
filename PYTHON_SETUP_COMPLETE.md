# Python Environment Setup - Complete

## What Was Done

### 1. Created Virtual Environment (venv_rag)
```powershell
cd C:\Users\STUDENT\Documents\vllm
& "C:\Users\STUDENT\AppData\Local\Programs\Python\Python39\python.exe" -m venv venv_rag
```
- Created a Python 3.9-based virtual environment at `venv_rag/`
- This isolates project dependencies from system Python

### 2. Created requirements.txt
- Location: `C:\Users\STUDENT\Documents\vllm\requirements.txt`
- Contains all necessary packages:
  - FastAPI & Uvicorn for server
  - LangChain ecosystem packages
  - SQLAlchemy & auth libraries  
  - Prometheus monitoring
  - All other dependencies

### 3. Installing Packages
- Running: `.\venv_rag\Scripts\python.exe -m pip install -r requirements.txt --no-cache-dir`
- This installs all dependencies into the isolated venv

### 4. Configured VS Code
- Updated `.vscode/settings.json`
- Set: `python.defaultInterpreterPath` → `${workspaceFolder}/venv_rag/Scripts/python.exe`
- VS Code will now use the venv by default

## How to Use

### Run RAG Server
```powershell
cd C:\Users\STUDENT\Documents\vllm\RAG
..\venv_rag\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 9096 --reload
```

### Install New Packages
```powershell
.\venv_rag\Scripts\pip.exe install <package_name>
```

### Run Python Scripts
```powershell
.\venv_rag\Scripts\python.exe script.py
```

## Troubleshooting

### If VS Code doesn't recognize the interpreter:
1. Reload VS Code (Ctrl+Shift+P → "Developer: Reload Window")
2. Select Python interpreter (Ctrl+Shift+P → "Python: Select Interpreter")
3. Choose the venv_rag interpreter

### If packages fail to install:
- Check internet connection
- Run: `.\venv_rag\Scripts\pip.exe list` to see installed packages
- Try reinstalling: `.\venv_rag\Scripts\pip.exe install -r requirements.txt --force-reinstall`

## Environment Details
- **Python Version**: 3.9.6
- **Virtual Environment**: venv_rag
- **Location**: C:\Users\STUDENT\Documents\vllm\venv_rag
- **Requirements File**: C:\Users\STUDENT\Documents\vllm\requirements.txt
