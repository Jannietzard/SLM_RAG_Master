"""
pytest conftest.py for logic_layer tests.

Adds the project root (Entwicklungfolder) to the Python path so that
'from src.logic_layer.X import Y' works regardless of the working directory.
"""
import sys
from pathlib import Path

# Project root = Entwicklungfolder (3 levels above this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
