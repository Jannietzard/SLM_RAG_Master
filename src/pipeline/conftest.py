"""
pytest conftest.py for pipeline tests.

Inserts the project root (Entwicklungfolder) onto sys.path so that
'from src.pipeline.X import Y' works regardless of the working directory
from which pytest is invoked.
"""
import sys
from pathlib import Path

# Project root = Entwicklungfolder (3 levels above this file)
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
assert PROJECT_ROOT.is_dir(), f"Expected project root at {PROJECT_ROOT}"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
