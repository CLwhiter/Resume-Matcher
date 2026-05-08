@echo off

start cmd /k "cd /d ""%~dp0apps\backend"" && uv run uvicorn app.main:app --reload --port 8000"

start cmd /k "cd /d ""%~dp0apps\frontend"" && npm run dev"
