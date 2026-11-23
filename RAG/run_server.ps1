# FastAPI RAG Server Startup Script
# This script runs the RAG server using Python 3.9

$pythonPath = "C:\Users\STUDENT\AppData\Local\Programs\Python\Python39\python.exe"
$scriptPath = Split-Path -Parent (Resolve-Path $MyInvocation.MyCommand.Path)

Set-Location $scriptPath
Write-Host "Starting RAG Server from: $scriptPath" -ForegroundColor Green
Write-Host "Using Python: $pythonPath" -ForegroundColor Yellow
Write-Host ""

& $pythonPath -m uvicorn server:app --host 0.0.0.0 --port 9096 --reload

pause
