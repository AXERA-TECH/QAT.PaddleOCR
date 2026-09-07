#!/usr/bin/env python3
"""Convert COCO-Text v2 (cocotext.v2.zip + train2014.zip) to PaddleOCR labels.

Inputs (under --dst):
  cocotext.v2.json         v2 annotation file (from cocotext.v2.zip)
  train2014/               COCO train2014 images (from train2014.zip)

Outputs:
  det_train.txt            word polygons (### kept as ignore)
  rec_train.txt            word crops (axis-aligned bbox + 2px margin)
  recrop/train/word_*.png  cropped word images

v2 schema (dict-based, NOT COCO arrays):
  imgs: {image_id: {id, set: "train"|"val", width, height, file_name}}
  anns: {ann_id: {id, image_id, bbox: [x,y,w,h], utf8_string, legibility,
                  language, class, area, mask: [x1,y1,...]}}
  imgToAnns: {image_id: [ann_id, ...]}

Only ``set == "train"`` images are converted (the val split's val2014 images
are not part of this download). ``mask`` (word polygon) is preferred for det;
``bbox`` is the fallback. Illegible/empty words become ``###`` ignore entries
with their valid polygon.
"""
from __future__ import annotations

import argparse
import json
import sys
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


def points_from_mask(mask):
    return [
        [float(mask[index]), float(mask[index + 1])]
        for index in range(0, len(mask) - 1, 2)
    ]


def points_from_bbox(bbox):
    x, y, w, h = (float(value) for value in bbox)
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = setup_common(parser.parse_args())

    annotation_path = args.dst / "cocotext.v2.json"
    image_root = args.dst / "train2014"
    if not annotation_path.exists():
        raise SystemExit(f"Missing annotations: {annotation_path}")
    if not image_root.is_dir():
        raise SystemExit(f"Missing images: {image_root}")

    with open(annotation_path, encoding="utf-8") as stream:
        data = json.load(stream)
    imgs = data["imgs"]
    anns = data["anns"]
    print(f"imgs: {len(imgs)}, anns: {len(anns)}", flush=True)

    train_imgs = {
        image_id: image
        for image_id, image in imgs.items()
        if image.get("set") == "train"
    }
    print(f"train imgs: {len(train_imgs)}", flush=True)

    recrop_dir = args.dst / "recrop" / "train"
    recrop_dir.mkdir(parents=True, exist_ok=True)

    det_rows = []
    rec_rows = []
    det_images = 0
    det_legible = 0
    det_ignored = 0
    rec_kept = 0
    rec_filtered = 0
    word_index = 0
    for image_id, image in train_imgs.items():
        file_name = Path(image["file_name"]).name
        relative = f"cocotext/train2014/{file_name}"
        if not (image_root / file_name).exists():
            continue
        det_entries = []
        for ann_id in data["imgToAnns"].get(image_id, []):
            annotation = anns.get(str(ann_id))
            if annotation is None:
                continue
            mask = annotation.get("mask") or []
            points = points_from_mask(mask) if len(mask) >= 6 else points_from_bbox(annotation["bbox"])
            if not validate_det_polygon(points):
                continue
            text = (annotation.get("utf8_string") or "").strip()
            if (
                annotation.get("legibility") != "legible"
                or not text
                or text == "###"
            ):
                det_entries.append({"transcription": "###", "points": points})
                det_ignored += 1
                continue
            det_entries.append({"transcription": text, "points": points})
            det_legible += 1
            cleaned = filter_rec_text(text, args.dictionary, args.max_text_length)
            if cleaned is None:
                rec_filtered += 1
                continue
            loaded = cv2.imread(str(image_root / file_name), cv2.IMREAD_COLOR)
            if loaded is None:
                rec_filtered += 1
                continue
            crop = crop_word(loaded, points)
            if crop is None or min(crop.shape[:2]) < args.min_side:
                rec_filtered += 1
                continue
            word_index += 1
            crop_name = f"word_{word_index:08d}.png"
            cv2.imwrite(str(recrop_dir / crop_name), crop)
            rec_rows.append((f"cocotext/recrop/train/{crop_name}", cleaned))
            rec_kept += 1
        det_rows.append((relative, det_entries))
        det_images += 1

    write_det_labels(args.dst / "det_train.txt", det_rows)
    write_rec_labels(args.dst / "rec_train.txt", rec_rows)
    print(
        f"det: {det_images} images, {det_legible} legible words, "
        f"{det_ignored} ignore(###) entries",
        flush=True,
    )
    print(f"rec: kept {rec_kept}, filtered {rec_filtered}", flush=True)
    print(f"outputs: {args.dst / 'det_train.txt'}, {args.dst / 'rec_train.txt'}", flush=True)


if __name__ == "__main__":
    main()
