@echo off

:: 启动后端 FastAPI（新窗口，进入backend目录）
start cmd /k "cd /d ""%~dp0apps\backend"" && uv run uvicorn app.main:app --reload --port 8000"

:: 启动前端 Next.js（新窗口，进入frontend目录）
start cmd /k "cd /d ""%~dp0apps\frontend"" && npm run dev"
