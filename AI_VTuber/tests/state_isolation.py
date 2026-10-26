"""Verhindert, dass Tests die echte data/luna_state.json verändern."""
import atexit
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.ai
import main
from core.luna_goals import LunaGoals
from core.luna_state import LunaState

_DIR = tempfile.mkdtemp(prefix="luna_state_test_")
atexit.register(shutil.rmtree, _DIR, ignore_errors=True)
_STATE = LunaState(Path(_DIR) / "luna_state.json")
_GOALS = LunaGoals(Path(_DIR) / "luna_goals.json")

main.get_luna_state = lambda: _STATE
core.ai.get_luna_state = lambda: _STATE
main.get_luna_goals = lambda: _GOALS
core.ai.get_luna_goals = lambda: _GOALS
