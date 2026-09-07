#!/usr/bin/env python3
"""Merge rec labels from multiple datasets into training/val label files.

Sources:
  train: cocotext/rec_train.txt (relative paths, cocotext/...)
         textocr/rec_train.txt  (absolute /tmp paths, local disk)
         icdr/rec_gt_train.txt  (rewrite prefix to icdr/...)
  val:   textocr/rec_val.txt    (absolute /tmp paths)
         (icdr rec_gt_test.txt is the ICDAR regression set, NOT merged here)

Optional image existence check per row (absolute paths stat directly; the
/tmp crops are local, the cocotext/icdr ones hit NFS attribute cache).
"""
from __future__ import annotations

import argparse
from pathlib import Path

# NOTE (2026-08-25): hiertext is converted but intentionally NOT merged
# into the training/validation sets for now (user decision).
TRAIN_SOURCES = [
    ("/home/heqi/dataset/cocotext/rec_train.txt", "", False),
    ("/home/heqi/dataset/textocr/rec_train.txt", "", False),
    ("/home/heqi/dataset/icdr/rec_gt_train.txt", "icdr/", False),
]
VAL_SOURCES = [
    ("/home/heqi/dataset/textocr/rec_val.txt", "", False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/home/heqi/dataset")
    parser.add_argument("--out-rec-train", default="/home/heqi/dataset/rec_train.txt")
    parser.add_argument("--out-rec-val", default="/home/heqi/dataset/rec_val.txt")
    parser.add_argument("--check-images", action="store_true", default=False)
    parser.add_argument("--no-check-images", action="store_true")
    args = parser.parse_args()
    if args.no_check_images:
        args.check_images = False
    data_dir = Path(args.data_dir)

    def process(sources, output):
        rows = 0
        missing = []
        lines_out = []
        for source, prefix, _fix in sources:
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
                        line = prefix + line
                    if args.check_images:
                        image_path = Path(line.split("\t", 1)[0])
                        full = image_path if image_path.is_absolute() else data_dir / image_path
                        if not full.exists():
                            missing.append(line.split("\t", 1)[0])
                            continue
                    lines_out.append(line)
                    count += 1
            rows += count
            print(f"  {source}: {count} rows", flush=True)
        Path(output).write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        print(f"total {rows} rows -> {output}", flush=True)
        if missing:
            print(f"  WARNING: {len(missing)} missing, first 10: {missing[:10]}", flush=True)
        return rows

    print("== train ==", flush=True)
    process(TRAIN_SOURCES, args.out_rec_train)
    print("== val ==", flush=True)
    process(VAL_SOURCES, args.out_rec_val)


if __name__ == "__main__":
    main()
