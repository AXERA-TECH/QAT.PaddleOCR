#!/usr/bin/env python3
"""Thin wrapper around tools/finetune_folded.py (quantized-domain fold).

The normalized CLI and reusable logic moved to:
    tools/finetune_folded.py
    pytorchocr/quantization/folding.py

This wrapper keeps the skill entry point working. Historical argument names
--checkpoint/--output map to --source-checkpoint/--fold-state-output.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

_MAPPING = {
    "--checkpoint": "--source-checkpoint",
    "--output": "--fold-state-output",
}


def _translate(argv):
    translated = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in _MAPPING:
            translated.append(_MAPPING[token])
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                translated.append(argv[index + 1])
                index += 2
                continue
        else:
            translated.append(token)
        index += 1
    return translated


def main():
    from tools.finetune_folded import main as cli_main

    sys.argv = ["finetune_folded.py", *_translate(sys.argv[1:])]
    cli_main()


if __name__ == "__main__":
    main()
