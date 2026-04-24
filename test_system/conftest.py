"""
pytest conftest.py for test_system/.

Adds the project root (Entwicklungfolder) to sys.path so that
'from src.data_layer.X import Y' and 'from src.logic_layer.X import Y'
work regardless of the working directory from which pytest is invoked.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT: Path = Path(__file__).parent.parent
assert PROJECT_ROOT.is_dir(), f"Expected project root at {PROJECT_ROOT}"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Exclude standalone diagnostic scripts that share the test_*.py naming
# convention but contain no pytest-collectible test functions.
# Pytest would silently report "no tests ran" for each, confusing the output.
# ---------------------------------------------------------------------------
collect_ignore_glob = [
    "diagnose_*.py",
]


# ---------------------------------------------------------------------------
# Skip graph-inspect tests on a clean machine where KuzuDB has not been
# populated yet.  Running ingestion first is a prerequisite.
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    graph_path = PROJECT_ROOT / "data" / "knowledge_graph_kuzu"
    if not graph_path.exists():
        skip_marker = pytest.mark.skip(
            reason="KuzuDB not populated — run 'python -X utf8 local_importingestion.py' first"
        )
        for item in items:
            if "graph_inspect" in str(item.fspath):
                item.add_marker(skip_marker)
