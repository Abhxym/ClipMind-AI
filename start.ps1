# AI Video Director - Quick Start Script (Windows PowerShell)
# Run this from the project root: .\start.ps1

Write-Host "`nAI Video Director - Starting..." -ForegroundColor Cyan

# Activate venv
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
    Write-Host "Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "No .venv found. Run: python -m venv .venv then pip install -r requirements.txt" -ForegroundColor Yellow
}

# Check .env
if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host ".env created from template. Please add your GOOGLE_API_KEY!" -ForegroundColor Yellow
}

$apiKey = (Get-Content .env | Where-Object { $_ -match "GOOGLE_API_KEY=(?!your)" })
if (-not $apiKey) {
    Write-Host ""
    Write-Host "------------------------------------------------" -ForegroundColor Red
    Write-Host "  ACTION REQUIRED: Set your Gemini API key" -ForegroundColor Red
    Write-Host "  1. Go to: https://aistudio.google.com/apikey" -ForegroundColor Red
    Write-Host "  2. Create an API key" -ForegroundColor Red
    Write-Host "  3. Open .env and set: GOOGLE_API_KEY=your_key" -ForegroundColor Red
    Write-Host "------------------------------------------------" -ForegroundColor Red
    Write-Host ""
}

# Start FastAPI backend in background
Write-Host "`nStarting FastAPI backend on http://localhost:8000..." -ForegroundColor Cyan
$backend = Start-Process python -ArgumentList "-m", "api.main" -PassThru -WindowStyle Minimized

# Start React frontend
Write-Host "Starting React dashboard on http://localhost:5173..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev" -WindowStyle Normal

Write-Host "`n------------------------------------------------" -ForegroundColor Green
Write-Host "  AI Video Director is running!" -ForegroundColor Green
Write-Host "  Dashboard:  http://localhost:5173" -ForegroundColor White
Write-Host "  API Docs:   http://localhost:8000/docs" -ForegroundColor White
Write-Host "  LangSmith:  https://smith.langchain.com" -ForegroundColor White
Write-Host "------------------------------------------------" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the backend" -ForegroundColor Gray

# Keep backend alive
Wait-Process -Id $backend.Id
