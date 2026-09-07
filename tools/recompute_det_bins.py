"""Recompute det task metrics from board/sim raw shrink-map bins.

Pairs `outputBin/<stem>/maps.bin` (float32 [1,1,640,640]) with the dataset
samples in label-file order and feeds them through the same validation metric
pipeline used by tools/evaluate_onnx.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pytorchocr.training import (
    build_dataset,
    build_validation_metric,
    detection_collate,
    load_ocr_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--bin-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--det-preprocess",
        choices=["letterbox", "paddle"],
        default="paddle",
        help=(
            "Detection resize preprocessing used to transform validation polygons; "
            "defaults to PaddleOCR fixed-shape resize."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_ocr_config(args.model_config)
    image_shape = tuple(config["Global"].get("d2s_train_image_shape", (3, 640, 640)))
    dataset = build_dataset(
        "det",
        args.model_config,
        config,
        image_shape,
        args.label_file,
        args.data_dir,
        return_polygons=True,
        det_preprocess=args.det_preprocess,
    )
    with open(args.label_file, encoding="utf-8") as handle:
        stems = [
            Path(line.split("\t")[0]).stem
            for line in handle
            if line.strip()
        ]
    if len(stems) != len(dataset):
        raise ValueError(f"label stems {len(stems)} != dataset {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        collate_fn=detection_collate,
    )
    metric = build_validation_metric("det", config, config_path=args.model_config)
    metric.reset()
    bin_root = Path(args.bin_root)
    seen = 0
    for (_images, targets), stem in zip(loader, stems):
        bin_path = bin_root / stem / "maps.bin"
        maps = np.fromfile(bin_path, dtype=np.float32).reshape(1, 1, 640, 640)
        metric.update(torch.from_numpy(maps), targets)
        seen += 1
    result = {
        "task": "det",
        "bin_root": str(bin_root.resolve()),
        "label_file": str(Path(args.label_file).resolve()),
        "sample_count": seen,
        **metric.compute(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
