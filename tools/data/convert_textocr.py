#!/usr/bin/env python3
"""Convert TextOCR (train_val_images.zip + TextOCR_0.1_*.json) to PaddleOCR labels.

Inputs (under --dst):
  images/train_images/            extracted images (train AND val share this
                                  directory; the json splits decide the set)
  annotations/TextOCR_0.1_{train,val,test}.json

Outputs:
  det_{train,val}.txt   word-level polygons (### kept as ignore)
  rec_{train,val}.txt   word crops
  recrop/{train,val}/word_{imgseq}_{wordseq}.png

Schema (dict-based, same family as cocotext v2):
  imgs: {image_id: {id, set, width, height, file_name: "train/xxx.jpg"}}
  anns: {ann_id: {id, image_id, bbox, utf8_string, points: [x1,y1,...], area}}
  imgToAnns: {image_id: [ann_id, ...]}

Parallelism: --split selects a single split (train or val) so independent
processes can run concurrently; --workers parallelizes per-image processing
(imread/imwrite release the GIL; crop PNGs are written by workers with
unique names, no lock needed). For NFS stalls, point --recrop-root at a
local disk and keep rec labels as absolute paths.
"""
from __future__ import annotations

import argparse
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

SPLIT_MAP = {"train": "train", "val": "val"}


def process_image(task):
    """Process one image; returns (relative, det_entries, rec_items)."""
    (
        image_id,
        image,
        image_seq,
        image_root,
        img_to_anns,
        anns,
        recrop_dir,
        recrop_root,
        args,
    ) = task
    file_name = Path(image["file_name"]).name
    relative = f"textocr/images/train_images/{file_name}"
    image_path = image_root / file_name
    if not image_path.exists():
        return None
    det_entries = []
    rec_items = []
    rec_local = 0
    rec_filtered = 0
    for ann_id in img_to_anns.get(image_id, []):
        annotation = anns.get(ann_id)
        if annotation is None:
            continue
        flat_points = annotation.get("points") or []
        points = [
            [float(flat_points[index]), float(flat_points[index + 1])]
            for index in range(0, len(flat_points) - 1, 2)
        ]
        if not validate_det_polygon(points):
            continue
        text = (annotation.get("utf8_string") or "").strip()
        if not text or text == "###":
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
        loaded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if loaded is None:
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
            crop_rel = f"textocr/recrop/{recrop_dir.name}/{crop_name}"
        else:
            crop_rel = str((recrop_dir / crop_name).resolve())
        rec_items.append((crop_rel, cleaned))
    return relative, det_entries, rec_items, rec_filtered


def convert_split(args, recrop_root, image_root, split):
    output_split = SPLIT_MAP[split]
    annotation_path = args.dst / "annotations" / f"TextOCR_0.1_{split}.json"
    if not annotation_path.exists():
        raise SystemExit(f"Missing annotations: {annotation_path}")
    recrop_dir = recrop_root / "recrop" / output_split
    if not args.det_only:
        recrop_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_path, encoding="utf-8") as stream:
        data = json.load(stream)
    imgs = data["imgs"]
    anns = data["anns"]
    img_to_anns = data.get("imgToAnns", {})

    tasks = [
        (
            image_id,
            image,
            image_seq,
            image_root,
            img_to_anns,
            anns,
            recrop_dir,
            recrop_root,
            args,
        )
        for image_seq, (image_id, image) in enumerate(imgs.items(), start=1)
    ]

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
        choices=("train", "val", "all"),
        default="all",
        help="Which official split to convert; run train and val in separate "
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

    image_root = args.dst / "images" / "train_images"
    if not image_root.is_dir():
        raise SystemExit(f"Missing images: {image_root} (unzip train_val_images.zip first)")

    splits = ("train", "val") if args.split == "all" else (args.split,)
    for split in splits:
        convert_split(args, recrop_root, image_root, split)


if __name__ == "__main__":
    main()
