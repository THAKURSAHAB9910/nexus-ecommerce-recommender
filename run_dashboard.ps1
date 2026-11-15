Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting NEXUS Recommendation Engine Dashboard" -ForegroundColor Green
Write-Host "  URL: http://localhost:8502" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan

python -m streamlit run app/streamlit_app.py --server.port 8502
