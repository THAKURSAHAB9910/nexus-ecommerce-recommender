@echo off
echo ========================================================
echo   Starting NEXUS FastAPI Production Backend
echo   Swagger UI: http://127.0.0.1:8000/docs
echo ========================================================
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
pause
