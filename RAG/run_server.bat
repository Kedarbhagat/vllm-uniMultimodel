@echo off
setlocal enabledelayedexpansion

set PYTHON_PATH=C:\Users\STUDENT\AppData\Local\Programs\Python\Python39\python.exe

echo Starting RAG Server...
echo Python: %PYTHON_PATH%
echo.

cd /d "%~dp0"

%PYTHON_PATH% -m uvicorn server:app --host 0.0.0.0 --port 9096 --reload

pause
