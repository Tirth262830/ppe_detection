"""Root Streamlit launcher for the PPE dashboard."""

from pathlib import Path
import runpy


APP_PATH = Path(__file__).parent / "app" / "dashboard.py"


if __name__ == "__main__":
    runpy.run_path(str(APP_PATH), run_name="__main__")
