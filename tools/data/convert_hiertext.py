#!/usr/bin/env python3
"""Convert HierText (images/*.tgz + annotations/*.jsonl.gz) to PaddleOCR labels.

Inputs (under --dst):
  images/{train,validation,test}/   extracted image splits
  annotations/{train,validation,test}.jsonl.gz

Outputs:
  det_{train,val}.txt   word-level polygons (### kept as ignore)
  rec_{train,val}.txt   word crops
  recrop/{train,val}/word_{imgseq}_{wordseq}.png

Split mapping: train -> train, validation -> val.  ``test`` is skipped (the
second evaluation track uses the official validation split).

Annotation schema (v1.0 single JSON object, COCO-style array):
  {"info": ..., "annotations": [
     {"image_id": "...", "image_width": ..., "image_height": ...,
      "paragraphs": [{"vertices": [[x,y],...], "legible": true,
                      "lines": [{"vertices": ..., "text": ...,
                                 "words": [{"vertices": [[x, y], ...],
                                            "text": ..., "legible": true,
                                            "handwritten": ..., "vertical": ...}]}]}]},
     ...]}

Det label rules (matching pytorchocr/training/data/det.py): every entry must
carry a >=3-point polygon; transcription "###" marks an ignore region, so
illegible/unreadable words are kept as "###" with their valid polygon.

Parallelism mirrors convert_textocr.py: --split runs one split per process,
--workers parallelizes per-image processing, --recrop-root points at a local
disk (rec labels then carry absolute crop paths) to avoid NFS small-file
stalls.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from convert_common import (
    add_common_args,
    crop_word,
    filter_rec_text,
    setup_common,
    validate_det_polygon,
    write_det_labels,
    write_rec_labels,
)

SPLIT_MAP = {"train": "train", "validation": "val"}


def collect_words(annotation):
    """Flatten paragraphs -> lines -> words from the v1.0 JSON format."""
    words = []
    for paragraph in annotation.get("paragraphs") or []:
        for line in paragraph.get("lines") or []:
            words.extend(line.get("words") or [])
    return words


def process_image(task):
    """One image -> (relative, det_entries, rec_items, rec_filtered) or None."""
    (
        image_relative,
        image_path,
        image_seq,
        words,
        recrop_dir,
        recrop_root,
        args,
    ) = task
    if not image_path.exists():
        return None
    loaded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if loaded is None:
        return None
    det_entries = []
    rec_items = []
    rec_local = 0
    rec_filtered = 0
    for word in words:
        vertices = word.get("vertices") or []
        points = [
            [float(vertex[0]), float(vertex[1])]
            for vertex in vertices
        ]
        if not validate_det_polygon(points):
            continue
        text = (word.get("text") or "").strip()
        if not word.get("legible", False) or not text or text == "###":
            if not args.rec_only:
                det_entries.append({"transcription": "###", "points": points})
            continue
        if not args.rec_only:
            det_entries.append({"transcription": text, "points": points})
        if args.det_only:
            continue
        cleaned = filter_rec_text(text, args.dictionary, args.max_text_length)
        if cleaned is None:
            rec_filtered += 1
            continue
        crop = crop_word(loaded, points)
        if crop is None or min(crop.shape[:2]) < args.min_side:
            rec_filtered += 1
            continue
        rec_local += 1
        crop_name = f"word_{image_seq:06d}_{rec_local:04d}.png"
        cv2.imwrite(str(recrop_dir / crop_name), crop)
        if str(recrop_root.resolve()) == str(args.dst.resolve()):
            crop_rel = f"hiertext/recrop/{recrop_dir.name}/{crop_name}"
        else:
            crop_rel = str((recrop_dir / crop_name).resolve())
        rec_items.append((crop_rel, cleaned))
    return image_relative, det_entries, rec_items, rec_filtered


def convert_split(args, recrop_root, image_root, split):
    output_split = SPLIT_MAP[split]
    annotation_path = args.dst / "annotations" / f"{split}.jsonl.gz"
    if not annotation_path.exists():
        raise SystemExit(f"Missing annotations: {annotation_path}")
    recrop_dir = recrop_root / "recrop" / output_split
    if not args.det_only:
        recrop_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    with gzip.open(annotation_path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    for image_seq, annotation in enumerate(data.get("annotations", []), start=1):
        image_id = annotation.get("image_id") or ""
        file_name = f"{image_id}.jpg"
        image_relative = f"hiertext/images/{split}/{file_name}"
        tasks.append(
            (
                image_relative,
                image_root / split / file_name,
                image_seq,
                collect_words(annotation),
                recrop_dir,
                recrop_root,
                args,
            )
        )

    det_rows = []
    rec_rows = []
    det_images = 0
    det_words = 0
    det_ignored = 0
    rec_kept = 0
    rec_filtered = 0

    def collect(result):
        nonlocal det_images, det_words, det_ignored, rec_kept, rec_filtered
        if result is None:
            return
        relative, det_entries, rec_items, filtered = result
        det_images += 1
        if not args.rec_only:
            det_words += sum(
                1 for entry in det_entries if entry["transcription"] != "###"
            )
            det_ignored += sum(
                1 for entry in det_entries if entry["transcription"] == "###"
            )
            det_rows.append((relative, det_entries))
        rec_kept += len(rec_items)
        rec_filtered += filtered
        rec_rows.extend(rec_items)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(process_image, tasks):
                collect(result)
    else:
        for task in tasks:
            collect(process_image(task))

    if not args.rec_only:
        write_det_labels(args.dst / f"det_{output_split}.txt", det_rows)
    if not args.det_only:
        write_rec_labels(args.dst / f"rec_{output_split}.txt", rec_rows)
    print(
        f"[{split}->{output_split}] det {det_images} images "
        f"({det_words} words, {det_ignored} ignore); "
        f"rec kept {rec_kept}, filtered {rec_filtered}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--det-only",
        action="store_true",
        help="Only produce det labels (no image IO, fast; skips rec crops).",
    )
    parser.add_argument(
        "--rec-only",
        action="store_true",
        help="Only produce rec crops/labels (skips det label writing).",
    )
    parser.add_argument(
        "--recrop-root",
        default=None,
        help="Root directory for recrop output (default: --dst). Use a local "
        "disk to avoid NFS small-file stalls; rec labels then carry absolute "
        "crop paths (RecognitionDataset accepts absolute paths).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "all"),
        default="all",
        help="Which official split to convert; run splits in separate "
        "processes to parallelize (default: all, sequential).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Per-image worker threads (imread/imwrite release the GIL). "
        "Default 1 (sequential).",
    )
    args = setup_common(parser.parse_args())
    if args.det_only and args.rec_only:
        raise SystemExit("--det-only and --rec-only are mutually exclusive.")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1.")
    recrop_root = Path(args.recrop_root) if args.recrop_root else args.dst

    image_root = args.dst / "images"
    splits = (
        ("train", "validation")
        if args.split == "all"
        else (args.split,)
    )
    for split in splits:
        convert_split(args, recrop_root, image_root, split)


if __name__ == "__main__":
    main()
