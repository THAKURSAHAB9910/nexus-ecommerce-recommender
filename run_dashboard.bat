@echo off
echo ========================================================
echo   Starting NEXUS Recommendation Engine Dashboard
echo   Host: http://localhost:8502
echo ========================================================
python -m streamlit run app/streamlit_app.py --server.port 8502
pause
