"""The benchmark entry point of pylbm (Python-HPC suite contract).

    python bench/run.py --quick   # < 2 min smoke run          -> results/quick.json
    python bench/run.py --dev     # < 1 GB GPU memory dev run   -> results/dev.json
    python bench/run.py --full    # < 15 min publishable run    -> results/rtx3070-laptop.json

Run ``--full`` on an otherwise idle GPU. ``--only triton cupy-cuda-c/fused-aa`` restricts the
backends.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pylbm.benchmark import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--dev", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    plan = "full" if a.full else "dev" if a.dev else "quick"
    run(plan, out=a.out, only=a.only)


if __name__ == "__main__":
    main()
