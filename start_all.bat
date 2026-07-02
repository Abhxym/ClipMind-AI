@echo off
echo ========================================================
echo Starting AI Video Director (Phase 2)
echo ========================================================

start "Backend API (Uvicorn)" cmd /k ".venv\Scripts\activate && uvicorn api.main:app --reload"

timeout /t 2 >nul

cd frontend
start "Frontend UI (Vite)" cmd /k "npm run dev"

echo Servers started!
echo Frontend should be available at http://localhost:5173
echo Backend API is running at http://localhost:8000
echo.
echo Ready for End-to-End Test!
