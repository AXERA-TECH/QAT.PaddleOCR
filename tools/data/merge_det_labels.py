#!/usr/bin/env python3
"""Merge det labels from multiple datasets into one training/val label file.

Inputs are PaddleOCR-format det labels (``image<TAB>[{...}]``). Sources:

  cocotext/det_train.txt      relative to --data-dir already (cocotext/...)
  textocr/det_{train,val}.txt relative already (textocr/...)
  icdr train/test labels      rewrite prefix to icdr/...
  ctw1500 training.txt        rewrite prefix to ctw1500/imgs/... and set
                              transcription "0" -> "###" (CTW1500 has no text)

Every referenced image is checked to exist under --data-dir before writing,
so training will not hit FileNotFoundError later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def rewrite_transcriptions(line: str, fix_ctw: bool) -> str:
    if not fix_ctw:
        return line
    image, payload = line.split("\t", 1)
    entries = json.loads(payload)
    for entry in entries:
        text = str(entry.get("transcription", ""))
        if text in ("0", ""):
            entry["transcription"] = "###"
    return f"{image}\t{json.dumps(entries, ensure_ascii=False)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/home/heqi/dataset")
    parser.add_argument("--out-det-train", required=True)
    parser.add_argument("--out-det-val", required=True)
    parser.add_argument("--check-images", action="store_true", default=True)
    parser.add_argument("--no-check-images", action="store_true")
    args = parser.parse_args()
    if args.no_check_images:
        args.check_images = False
    data_dir = Path(args.data_dir)

    # (source file, path prefix to prepend, ctw transcription fix)
    # NOTE (2026-08-25): hiertext is converted but intentionally NOT merged
    # into the training/validation sets for now (user decision).
    TRAIN_SOURCES = [
        ("/home/heqi/dataset/cocotext/det_train.txt", "", False),
        ("/home/heqi/dataset/textocr/det_train.txt", "", False),
        ("/home/heqi/dataset/icdr/train_icdar2015_label.txt", "icdr/", False),
        ("/home/heqi/dataset/ctw1500/imgs/training.txt", "ctw1500/imgs/", True),
    ]
    VAL_SOURCES = [
        ("/home/heqi/dataset/textocr/det_val.txt", "", False),
        ("/home/heqi/dataset/icdr/test_icdar2015_label.txt", "icdr/", False),
    ]

    def process(sources, output):
        rows = 0
        missing = []
        lines_out = []
        for source, prefix, fix_ctw in sources:
            source_path = Path(source)
            if not source_path.exists():
                raise SystemExit(f"Missing source: {source_path}")
            count = 0
            with open(source_path, encoding="utf-8") as stream:
                for line in stream:
                    line = line.rstrip("\r\n")
                    if not line.strip():
                        continue
                    image, _ = line.split("\t", 1)
                    if prefix:
                        image = prefix + image
                        line = image + "\t" + line.split("\t", 1)[1]
                    if fix_ctw:
                        line = rewrite_transcriptions(line, True)
                    if args.check_images:
                        image_path = Path(image)
                        full = image_path if image_path.is_absolute() else data_dir / image_path
                        if not full.exists():
                            missing.append(image)
                            continue
                    lines_out.append(line)
                    count += 1
            rows += count
            print(f"  {source}: {count} rows", flush=True)
        Path(output).write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        print(f"total {rows} rows -> {output}", flush=True)
        if missing:
            print(f"  WARNING: {len(missing)} missing images, first 10: {missing[:10]}", flush=True)
        return rows

    print("== train ==", flush=True)
    process(TRAIN_SOURCES, args.out_det_train)
    print("== val ==", flush=True)
    process(VAL_SOURCES, args.out_det_val)


if __name__ == "__main__":
    main()
