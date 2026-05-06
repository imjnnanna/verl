"""Bypass verl/__init__.py's heavy imports (ray, torch, etc.) for simu unit tests.

The simu subpackage has no dependencies on the rest of verl, so we register
empty stub packages for `verl`, `verl.trainer`, `verl.trainer.ppo` and
`verl.trainer.ppo.simu` before pytest imports the test module. The stubs
have correct `__path__` entries so subsequent absolute imports of
`verl.trainer.ppo.simu.<x>` resolve normally.
"""
from __future__ import annotations
import sys
import types
from pathlib import Path

_simu_dir = Path(__file__).resolve().parent.parent
_ppo_dir = _simu_dir.parent
_trainer_dir = _ppo_dir.parent
_verl_dir = _trainer_dir.parent

for pkg_name, pkg_path in [
    ("verl", _verl_dir),
    ("verl.trainer", _trainer_dir),
    ("verl.trainer.ppo", _ppo_dir),
    ("verl.trainer.ppo.simu", _simu_dir),
]:
    if pkg_name not in sys.modules or not hasattr(sys.modules[pkg_name], "__path__"):
        mod = types.ModuleType(pkg_name)
        mod.__path__ = [str(pkg_path)]
        sys.modules[pkg_name] = mod
