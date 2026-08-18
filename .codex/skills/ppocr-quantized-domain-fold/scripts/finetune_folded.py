#!/usr/bin/env python3
"""Thin wrapper around tools/finetune_folded.py (folded-model finetune).

The normalized CLI and reusable logic moved to:
    tools/finetune_folded.py
    pytorchocr/quantization/folding.py

This wrapper keeps the skill entry point working. Historical argument name
--source-checkpoint is unchanged; --folded-state is unchanged.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))


def main():
    from tools.finetune_folded import main as cli_main

    sys.argv = ["finetune_folded.py", *sys.argv[1:]]
    cli_main()


if __name__ == "__main__":
    main()
