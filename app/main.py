"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Entrypoint: app/main.py (Alias for app/streamlit_app.py)
"""
import runpy
from pathlib import Path

_target = Path(__file__).parent / "streamlit_app.py"
runpy.run_path(str(_target), run_name="__main__")
