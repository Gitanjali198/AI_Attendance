@echo off
title AI Attendance System - FastAPI REST Backend
echo ========================================================
echo    Launching FastAPI REST API (Swagger UI: /docs)
echo ========================================================
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
pause
