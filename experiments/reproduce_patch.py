"""Run the draft-patch variant on the full validation set.

Reuses :mod:`experiments.reproduce_zarutskiy`'s machinery and swaps only the
algorithm source, output directory and experiment id.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import experiments.reproduce_zarutskiy as R

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "clarify/algorithms/draft_patch_clarifier.py"
OUTPUT = ROOT / "runs" / "patch_val"
EXPERIMENT = "patch_val"
SOURCE_SHA = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    R.SOURCE = SOURCE
    R.OUTPUT = OUTPUT
    R.EXPERIMENT = EXPERIMENT
    R.SOURCE_SHA = SOURCE_SHA
    R.COMMIT = COMMIT
    R.main()


if __name__ == "__main__":
    main()
