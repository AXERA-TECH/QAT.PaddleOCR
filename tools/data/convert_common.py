#!/usr/bin/env python3
"""Shared helpers for converting large-scale datasets to PaddleOCR format.

Label formats produced (matching pytorchocr/training/data/{det,rec}.py):

- det: ``<image_path>\t[{"transcription": "...", "points": [[x,y],...]}, ...]``
  (``###`` entries are kept for ignore regions, matching PaddleOCR det labels)
- rec: ``<image_path>\t<text>`` where image_path points to a cropped word image

Filtering follows the CTCLabelEncoder/NRTRLabelEncoder contracts:
- text must be non-empty and not contain ``###`` (rec only; det keeps ``###``)
- at least one character must be in the dictionary
- dictionary-character count must be <= max_text_length - 2 (NRTR keeps
  len(encoded) < max_text_length - 1, CTC keeps len(text) <= max_text_length)
- polygon must have >= 3 points and non-zero area
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_dictionary(path: str) -> set[str]:
    with open(path, encoding="utf-8") as stream:
        return {line.rstrip("\r\n") for line in stream}


def filter_rec_text(text: str, dictionary: set[str], max_text_length: int):
    """Return the cleaned text if encodable by both CTC and NRTR, else None."""
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned or "###" in cleaned:
        return None
    if len(cleaned) > max_text_length:
        return None
    in_dict = [char for char in cleaned if char in dictionary]
    if not in_dict:
        return None
    if len(in_dict) > max_text_length - 2:
        return None
    return cleaned


def polygon_area(points) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def validate_det_polygon(points) -> bool:
    if len(points) < 3:
        return False
    try:
        if polygon_area(points) <= 0.0:
            return False
    except (TypeError, ValueError):
        return False
    return True


def bbox_of_polygon(points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def crop_word(image, points, margin: int = 2):
    """Axis-aligned word crop with a small margin, clamped to the image."""
    import cv2

    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox_of_polygon(points)
    x_min = max(0, int(x_min) - margin)
    y_min = max(0, int(y_min) - margin)
    x_max = min(width - 1, int(x_max) + margin)
    y_max = min(height - 1, int(y_max) + margin)
    if x_max - x_min < 1 or y_max - y_min < 1:
        return None
    return image[y_min : y_max + 1, x_min : x_max + 1]


def write_det_labels(path: Path, rows):
    """rows: list of (image_rel_path, list of dict entries)."""
    with open(path, "w", encoding="utf-8") as stream:
        for image_path, entries in rows:
            payload = json.dumps(entries, ensure_ascii=False)
            stream.write(f"{image_path}\t{payload}\n")


def write_rec_labels(path: Path, rows):
    """rows: list of (image_rel_path, text)."""
    with open(path, "w", encoding="utf-8") as stream:
        for image_path, text in rows:
            stream.write(f"{image_path}\t{text}\n")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dst", required=True, help="Dataset root (e.g. /home/heqi/dataset/cocotext).")
    parser.add_argument(
        "--data-dir",
        default="/home/heqi/dataset",
        help="Base directory label image paths are relative to.",
    )
    parser.add_argument(
        "--dict",
        default="pytorchocr/utils/dict/ppocrv6_dict.txt",
        help="Character dictionary for rec filtering.",
    )
    parser.add_argument("--max-text-length", type=int, default=25)
    parser.add_argument("--min-side", type=int, default=2, help="Minimum crop side in px.")


def setup_common(args):
    args.dst = Path(args.dst)
    args.data_dir = str(Path(args.data_dir).resolve())
    args.dictionary = load_dictionary(args.dict)
    return args
