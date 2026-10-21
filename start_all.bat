@echo off
echo ========================================================
echo Starting ClipMind AI
echo ========================================================

start "Backend API (Uvicorn)" cmd /k ".venv\Scripts\activate && uvicorn api.main:app --reload"

timeout /t 2 >nul

cd frontend
if not exist node_modules call npm install
start "Frontend UI (Vite)" cmd /k "npm run dev"
cd ..

echo Servers started!
echo Frontend should be available at http://localhost:5173
echo Backend API is running at http://localhost:8000
echo API docs at http://localhost:8000/docs
