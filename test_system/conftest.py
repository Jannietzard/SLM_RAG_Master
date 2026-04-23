"""
pytest conftest.py for test_system/.

Adds the project root (Entwicklungfolder) to sys.path so that
'from src.data_layer.X import Y' and 'from src.logic_layer.X import Y'
work regardless of the working directory from which pytest is invoked.
"""
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).parent.parent
assert PROJECT_ROOT.is_dir(), f"Expected project root at {PROJECT_ROOT}"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
