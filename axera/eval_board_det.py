"""Board-side PP-OCR detection axmodel inference (AX650, axengine).

Feeds the same FP32 NCHW blob contract as the QuantONNX/ORT eval. By default,
the image is resized directly to 640x640 to match PaddleOCR's fixed-shape
preprocessing. The historical centered letterbox path can be selected with
``--det-preprocess letterbox``. Saves raw float32 shrink maps to
``--output-dir/<stem>/maps.bin``; DB postprocess and hmean are computed offline
on the host.

Optional ``--vis-dir`` runs the DB postprocess on the board and writes one
overlay per sample (``<vis-dir>/<stem>.jpg``, red boxes, optional score text),
so detections can be inspected without a host round trip. Box extraction
mirrors ``pytorchocr/postprocess/db_postprocess.py`` (same ``get_mini_boxes``
ordering, ``box_score_fast`` scoring and min-side 3 / 3+2 filters); only the
pyclipper/shapely ``unclip`` is replaced by an equivalent rotated-rect
expansion because neither dependency exists on the board. Threshold defaults
follow the v6 det config (``configs/det/PP-OCRv6/PP-OCRv6_small_det.yml``:
thresh 0.2, box_thresh 0.45, unclip_ratio 1.4, max_candidates 3000) and can be
overridden per run. Pass ``--no-maps`` when only overlays are wanted.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

TARGET = 640
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# DB filters from pytorchocr/postprocess/db_postprocess.py
MIN_SIZE = 3
MIN_SIZE_AFTER_UNCLIP = MIN_SIZE + 2

# v6 det deploy config (configs/det/PP-OCRv6/PP-OCRv6_small_det.yml)
VIS_THRESH = 0.2
VIS_BOX_THRESH = 0.45
VIS_UNCLIP_RATIO = 1.4
VIS_MAX_CANDIDATES = 3000


class DetGeometry(NamedTuple):
    """Target(640x640) to source mapping for one detection preprocessing mode.

    ``src_x = (dst_x - left) * scale_x`` and ``src_y = (dst_y - top) * scale_y``.
    Direct resize uses independent axis scales because the aspect ratio is not
    preserved; letterbox scales uniformly and centers the resized content.
    """

    resize_w: int
    resize_h: int
    left: int
    top: int
    scale_x: float
    scale_y: float


def det_geometry(height, width, det_preprocess="paddle"):
    """Return the target-to-source mapping used by :func:`preprocess`."""
    if det_preprocess == "paddle":
        return DetGeometry(TARGET, TARGET, 0, 0, width / TARGET, height / TARGET)
    if det_preprocess == "letterbox":
        scale = min(TARGET / width, TARGET / height)
        resize_w = min(TARGET, max(1, round(width * scale)))
        resize_h = min(TARGET, max(1, round(height * scale)))
        return DetGeometry(
            resize_w,
            resize_h,
            (TARGET - resize_w) // 2,
            (TARGET - resize_h) // 2,
            1.0 / scale,
            1.0 / scale,
        )
    raise ValueError("Detection preprocessing must be 'paddle' or 'letterbox'.")


def preprocess(img, det_preprocess="paddle"):
    geometry = det_geometry(*img.shape[:2], det_preprocess)
    if det_preprocess == "paddle":
        img = cv2.resize(img, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(
            img, (geometry.resize_w, geometry.resize_h), interpolation=cv2.INTER_LINEAR
        )
        img = cv2.copyMakeBorder(
            img,
            geometry.top,
            TARGET - geometry.resize_h - geometry.top,
            geometry.left,
            TARGET - geometry.resize_w - geometry.left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
    img = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(((img - MEAN) / STD).transpose(2, 0, 1)[None])


def get_mini_boxes(contour):
    """Ordered rotated rect of a contour (same ordering as DBPostProcess)."""
    if len(contour) < 5:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return np.array(
                [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]], dtype=np.float32
            ), 0.0
        return np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
        ), float(min(w, h))

    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda point: point[0])
    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2
    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return np.array(box, dtype=np.float32), float(min(bounding_box[1]))


def box_score_fast(shrink, box):
    """Mean shrink-map probability inside a box (DBPostProcess.box_score_fast)."""
    height, width = shrink.shape[:2]
    box = np.asarray(box, dtype=np.float32).copy()
    xmin = int(np.clip(np.floor(box[:, 0].min()), 0, width - 1))
    xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, width - 1))
    ymin = int(np.clip(np.floor(box[:, 1].min()), 0, height - 1))
    ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, height - 1))

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(shrink[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def unclip_box(box, unclip_ratio):
    """Expand a rotated-rect box outwards by ``area * ratio / perimeter``.

    DBPostProcess offsets the polygon with pyclipper using exactly that
    distance; on a rectangle it is an expansion of every side by ``distance``,
    so the result is rebuilt from ``minAreaRect`` here instead of pulling
    pyclipper/shapely onto the board.
    """
    (center_x, center_y), (width, height), angle = cv2.minAreaRect(
        np.asarray(box, dtype=np.float32)
    )
    perimeter = 2.0 * (width + height)
    if perimeter <= 0:
        return np.asarray(box, dtype=np.float32)
    distance = (width * height) * unclip_ratio / perimeter
    return cv2.boxPoints(
        ((center_x, center_y), (width + 2.0 * distance, height + 2.0 * distance), angle)
    ).astype(np.float32)


def detect_boxes(
    shrink,
    thresh=VIS_THRESH,
    box_thresh=VIS_BOX_THRESH,
    unclip_ratio=VIS_UNCLIP_RATIO,
    max_candidates=VIS_MAX_CANDIDATES,
):
    """Extract DB boxes (640x640 coordinates) and scores from a shrink map."""
    shrink = np.asarray(shrink, dtype=np.float32)
    found = cv2.findContours(
        (shrink > thresh).astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = found[1] if len(found) == 3 else found[0]

    boxes, scores = [], []
    for contour in contours[:max_candidates]:
        contour = contour.reshape(-1, 2)
        if len(contour) < 3:
            continue
        points, min_side = get_mini_boxes(contour)
        if min_side < MIN_SIZE:
            continue
        score = box_score_fast(shrink, points)
        if score < box_thresh:
            continue
        expanded, min_side = get_mini_boxes(unclip_box(points, unclip_ratio))
        if min_side < MIN_SIZE_AFTER_UNCLIP:
            continue
        boxes.append(expanded)
        scores.append(score)
    return boxes, scores


def map_boxes_to_source(boxes, geometry, src_height, src_width):
    """Map 640x640 box coordinates back to the original image."""
    mapped = []
    for box in boxes:
        box = np.asarray(box, dtype=np.float32)
        out = np.empty_like(box)
        out[:, 0] = np.clip(
            np.round((box[:, 0] - geometry.left) * geometry.scale_x), 0, src_width
        )
        out[:, 1] = np.clip(
            np.round((box[:, 1] - geometry.top) * geometry.scale_y), 0, src_height
        )
        mapped.append(out.astype(np.int32))
    return mapped


def draw_boxes(img, boxes, scores=None, thickness=2, show_score=False):
    """Return a copy of ``img`` with the boxes (and optional scores) drawn."""
    canvas = img.copy()
    for index, box in enumerate(boxes):
        cv2.polylines(canvas, [np.asarray(box, dtype=np.int32).reshape(-1, 1, 2)],
                      True, (0, 0, 255), thickness)
        if show_score and scores is not None:
            left = int(np.asarray(box)[:, 0].min())
            top = int(np.asarray(box)[:, 1].min())
            cv2.putText(
                canvas,
                f"{scores[index]:.2f}",
                (left, max(12, top - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return canvas


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
    parser.add_argument(
        "--maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write raw shrink maps to --output-dir/<stem>/maps.bin (--no-maps to skip).",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="Save one overlay per sample (<dir>/<stem>.jpg) with DB boxes; disabled by default.",
    )
    parser.add_argument("--vis-thresh", type=float, default=VIS_THRESH,
                        help=f"DB binarization threshold (default {VIS_THRESH}, v6 det config)")
    parser.add_argument("--vis-box-thresh", type=float, default=VIS_BOX_THRESH,
                        help=f"Minimum box score (default {VIS_BOX_THRESH}, v6 det config)")
    parser.add_argument("--vis-unclip-ratio", type=float, default=VIS_UNCLIP_RATIO,
                        help=f"DB unclip ratio (default {VIS_UNCLIP_RATIO}, v6 det config)")
    parser.add_argument("--vis-max-candidates", type=int, default=VIS_MAX_CANDIDATES,
                        help=f"Max contours per image (default {VIS_MAX_CANDIDATES})")
    parser.add_argument("--vis-score", action="store_true", help="Draw per-box scores")
    parser.add_argument("--vis-thickness", type=int, default=2, help="Box line thickness")
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
    if args.maps:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
        print(
            f"visualization: {args.vis_dir} (thresh={args.vis_thresh}, "
            f"box_thresh={args.vis_box_thresh}, unclip_ratio={args.vis_unclip_ratio})",
            flush=True,
        )
    start = time.perf_counter()
    total_boxes = 0
    for index, (image_name, stem) in enumerate(zip(image_names, stems)):
        img_path = os.path.join(args.image_dir, os.path.basename(image_name))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot decode {img_path}")
        blob = preprocess(img, args.det_preprocess)
        output = session.run(None, {input_name: blob})[0]
        if args.maps:
            out_dir = os.path.join(args.output_dir, stem)
            os.makedirs(out_dir, exist_ok=True)
            np.asarray(output, dtype=np.float32).tofile(os.path.join(out_dir, "maps.bin"))
        if args.vis_dir:
            shrink = np.asarray(output, dtype=np.float32)
            shrink = shrink.reshape(shrink.shape[-2], shrink.shape[-1])
            boxes, scores = detect_boxes(
                shrink,
                thresh=args.vis_thresh,
                box_thresh=args.vis_box_thresh,
                unclip_ratio=args.vis_unclip_ratio,
                max_candidates=args.vis_max_candidates,
            )
            geometry = det_geometry(*img.shape[:2], args.det_preprocess)
            boxes = map_boxes_to_source(boxes, geometry, img.shape[0], img.shape[1])
            overlay = draw_boxes(
                img, boxes, scores, args.vis_thickness, args.vis_score
            )
            cv2.imwrite(os.path.join(args.vis_dir, f"{stem}.jpg"), overlay)
            total_boxes += len(boxes)
        if args.progress_every and (index + 1) % args.progress_every == 0:
            elapsed = time.perf_counter() - start
            print(f"{index + 1}/{len(stems)} done, {elapsed:.1f}s", flush=True)
    elapsed = time.perf_counter() - start
    if args.vis_dir:
        print(
            f"all done in {elapsed:.1f}s, {total_boxes} boxes over {len(stems)} images",
            flush=True,
        )
    else:
        print(f"all done in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
