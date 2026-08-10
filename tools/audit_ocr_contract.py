import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pytorchocr.diagnostics import (
    audit_detection_label,
    audit_recognition_label,
    json_hash,
    load_characters,
    model_contract,
    validate_audit,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze PP-OCR detection/recognition accuracy contracts."
    )
    parser.add_argument("--det-model-config", required=True)
    parser.add_argument("--det-paddle-weights", required=True)
    parser.add_argument("--det-torch-weights", required=True)
    parser.add_argument("--det-train-label", required=True)
    parser.add_argument("--det-val-label", required=True)
    parser.add_argument("--det-data-dir", required=True)
    parser.add_argument("--rec-model-config", required=True)
    parser.add_argument("--rec-paddle-weights", required=True)
    parser.add_argument("--rec-torch-weights", required=True)
    parser.add_argument("--rec-train-label", required=True)
    parser.add_argument("--rec-val-label", required=True)
    parser.add_argument("--rec-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--det-debug-samples", type=int, default=16)
    parser.add_argument("--rec-debug-samples", type=int, default=32)
    return parser.parse_args()


def package_version(module_name):
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as error:
        return f"unavailable: {type(error).__name__}"


def git_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_debug_samples(path, samples):
    lines = [
        f'{item["image"]}\t{json.dumps(item, ensure_ascii=False, sort_keys=True)}'
        for item in samples
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    det_model = model_contract(
        "det",
        args.det_model_config,
        args.det_paddle_weights,
        args.det_torch_weights,
    )
    rec_model = model_contract(
        "rec",
        args.rec_model_config,
        args.rec_paddle_weights,
        args.rec_torch_weights,
    )
    rec_characters = load_characters(
        rec_model["dictionary_path"],
        rec_model["use_space_char"],
    )
    det_audit = {
        "model": det_model,
        "postprocess": {
            "box_thresh": 0.6,
            "box_type": "quad",
            "iou_constraint": 0.5,
            "max_candidates": 1000,
            "score_mode": "fast",
            "thresh": 0.3,
            "unclip_ratio": 1.5,
        },
        "preprocessing": {
            "color_order": "BGR",
            "deploy_image_shape": det_model["image_shape"],
            "mean": [0.485, 0.456, 0.406],
            "padding_value": 114,
            "std": [0.229, 0.224, 0.225],
        },
        "train": audit_detection_label(
            args.det_train_label,
            args.det_data_dir,
            args.det_debug_samples,
        ),
        "val": audit_detection_label(
            args.det_val_label,
            args.det_data_dir,
            args.det_debug_samples,
        ),
    }
    rec_audit = {
        "decoder": {
            "blank_index": 0,
            "ignore_space_for_metric": True,
            "remove_duplicates": True,
        },
        "model": rec_model,
        "preprocessing": {
            "color_order": "BGR",
            "image_shape": rec_model["image_shape"],
            "normalize": "(x / 255 - 0.5) / 0.5",
            "padding_side": "right",
            "padding_value_after_normalize": 0.0,
        },
        "train": audit_recognition_label(
            args.rec_train_label,
            args.rec_data_dir,
            rec_characters,
            rec_model["max_text_length"],
            args.rec_debug_samples,
        ),
        "val": audit_recognition_label(
            args.rec_val_label,
            args.rec_data_dir,
            rec_characters,
            rec_model["max_text_length"],
            args.rec_debug_samples,
        ),
    }
    common = {
        "environment": {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cv2": cv2.__version__,
            "numpy": np.__version__,
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime"),
            "paddle": package_version("paddle"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "shapely": package_version("shapely"),
            "torch": package_version("torch"),
        },
        "git": git_state(),
        "project_root": str(ROOT_DIR),
    }
    common["contract_sha256"] = json_hash(
        {
            "environment": common["environment"],
            "git_commit": common["git"].get("commit"),
            "project_root": common["project_root"],
        }
    )
    det_audit["contract_sha256"] = json_hash(det_audit)
    rec_audit["contract_sha256"] = json_hash(rec_audit)
    write_json(output_dir / "common.json", common)
    write_json(output_dir / "det.json", det_audit)
    write_json(output_dir / "rec.json", rec_audit)
    write_debug_samples(
        output_dir / "det_debug_samples.txt",
        det_audit["val"]["debug_samples"],
    )
    write_debug_samples(
        output_dir / "rec_debug_samples.txt",
        rec_audit["val"]["debug_samples"],
    )
    failures = validate_audit(det_audit, "det") + validate_audit(rec_audit, "rec")
    summary = {
        "contracts": {
            "common": common["contract_sha256"],
            "det": det_audit["contract_sha256"],
            "rec": rec_audit["contract_sha256"],
        },
        "det_samples": {
            "train": det_audit["train"]["sample_count"],
            "val": det_audit["val"]["sample_count"],
        },
        "failures": failures,
        "rec_samples": {
            "train": rec_audit["train"]["sample_count"],
            "train_loader_accepted": rec_audit["train"]["loader_accepted_samples"],
            "train_strictly_encodable": rec_audit["train"]["strictly_encodable_samples"],
            "val": rec_audit["val"]["sample_count"],
            "val_loader_accepted": rec_audit["val"]["loader_accepted_samples"],
            "val_strictly_encodable": rec_audit["val"]["strictly_encodable_samples"],
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main(parse_args())
