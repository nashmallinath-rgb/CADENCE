"""Run provider smoke checks. Prefer: python -m cadence.cli check-setup"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadence.check_setup import run_check_setup


if __name__ == "__main__":
    sys.exit(run_check_setup())
