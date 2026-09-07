"""Board-side PP-OCR detection axmodel inference (AX650, axengine).

Feeds the same FP32 NCHW blob contract as the QuantONNX/ORT eval. By default,
the image is resized directly to 640x640 to match PaddleOCR's fixed-shape
preprocessing. The historical centered letterbox path can be selected with
``--det-preprocess letterbox``. Saves raw float32 shrink maps to
``--output-dir/<stem>/maps.bin``; DB postprocess and hmean are computed offline
on the host.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

TARGET = 640
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img, det_preprocess="paddle"):
    h, w = img.shape[:2]
    if det_preprocess == "paddle":
        img = cv2.resize(img, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)
    elif det_preprocess == "letterbox":
        scale = min(TARGET / w, TARGET / h)
        rw = min(TARGET, max(1, round(w * scale)))
        rh = min(TARGET, max(1, round(h * scale)))
        img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        left = (TARGET - rw) // 2
        top = (TARGET - rh) // 2
        img = cv2.copyMakeBorder(
            img,
            top,
            TARGET - rh - top,
            left,
            TARGET - rw - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
    else:
        raise ValueError(
            "Detection preprocessing must be 'paddle' or 'letterbox'."
        )
    img = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(((img - MEAN) / STD).transpose(2, 0, 1)[None])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axmodel", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-name", default=None)
    parser.add_argument(
        "--det-preprocess",
        choices=["paddle", "letterbox"],
        default="paddle",
        help=(
            "Detection preprocessing; defaults to PaddleOCR fixed-shape resize. "
            "Use letterbox only for the historical centered-padding contract."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all labels")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    import axengine

    stems = []
    image_names = []
    with open(args.label_file, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            name = line.split("\t", 1)[0]
            image_names.append(name)
            stems.append(Path(name).stem)
    if args.max_samples:
        image_names = image_names[:args.max_samples]
        stems = stems[:args.max_samples]
    print(f"total samples: {len(stems)}", flush=True)

    session = axengine.InferenceSession(args.axmodel)
    input_name = args.input_name or session.get_inputs()[0].name
    os.makedirs(args.output_dir, exist_ok=True)
    start = time.perf_counter()
    for index, (image_name, stem) in enumerate(zip(image_names, stems)):
        img_path = os.path.join(args.image_dir, os.path.basename(image_name))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot decode {img_path}")
        blob = preprocess(img, args.det_preprocess)
        output = session.run(None, {input_name: blob})[0]
        out_dir = os.path.join(args.output_dir, stem)
        os.makedirs(out_dir, exist_ok=True)
        np.asarray(output, dtype=np.float32).tofile(os.path.join(out_dir, "maps.bin"))
        if args.progress_every and (index + 1) % args.progress_every == 0:
            elapsed = time.perf_counter() - start
            print(f"{index + 1}/{len(stems)} done, {elapsed:.1f}s", flush=True)
    print(f"all done in {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
