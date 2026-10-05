import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(ROOT / "src"))
FIX = Path(__file__).parent / "fixtures"


def load_fixture(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


@pytest.fixture
def history_14():
    turns = []
    for i in range(1, 8):
        turns += [{"speaker": "user", "text": f"u{i}"}, {"speaker": "agent", "text": f"a{i}"}]
    return turns
