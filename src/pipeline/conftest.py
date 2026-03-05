"""
pytest conftest.py für pipeline Tests.

Fügt den Projekt-Root (Entwicklungfolder) zum Python-Pfad hinzu,
damit 'from src.pipeline.X import Y' unabhängig vom Ausführungsort funktioniert.
"""
import sys
from pathlib import Path

# Projekt-Root = Entwicklungfolder (3 Ebenen über diesem File)
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
