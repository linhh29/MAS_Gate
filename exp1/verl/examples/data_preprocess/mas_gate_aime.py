#!/usr/bin/env python3
"""Compatibility wrapper. MAS-Gate implementation lives in mas_gate_rl."""
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VERL_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT, _VERL_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mas_gate_rl.data_preprocess import main

if __name__ == "__main__":
    main()
