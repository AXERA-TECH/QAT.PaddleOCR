"""Compare a PP-OCR detection QuantONNX with a compiled axmodel (simulation).

Preprocessing exactly matches pytorchocr/training/data/det.py eval contract:
BGR (no RGB swap) -> center letterbox to 640x640 (pad 114) -> /255 ->
ImageNet mean/std -> NCHW float32. The same float32 blob feeds both ORT and
the axmodel (config declares src_dtype FP32, mean 0, std 1).

The script must run on a Pulsar2 host with ``pulsar2 run`` available.
"""
import argparse
import glob
import os
import subprocess
import sys

import cv2
import numpy as np
import onnxruntime as ort

TARGET_H = TARGET_W = 640
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def onnx_type_to_np_dtype(onnx_type_str):
    core = onnx_type_str[len("tensor("):-1].strip()
    mapping = {
        "int64": np.int32, "int32": np.int32, "int16": np.int16,
        "int8": np.int8, "uint64": np.uint32, "uint32": np.uint32,
        "uint16": np.uint16, "uint8": np.uint8,
        "float": np.float32, "float32": np.float32, "float64": np.float64,
        "bool": np.bool_,
    }
    if core not in mapping:
        raise ValueError(f"unsupported ONNX type: {onnx_type_str}")
    return mapping[core]


def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR, matches training data pipeline
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    scale = min(TARGET_W / w, TARGET_H / h)
    rw = min(TARGET_W, max(1, round(w * scale)))
    rh = min(TARGET_H, max(1, round(h * scale)))
    img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    left = (TARGET_W - rw) // 2
    right = TARGET_W - rw - left
    top = (TARGET_H - rh) // 2
    bottom = TARGET_H - rh - top
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = img.astype(np.float32) / 255.0
    img = ((img - MEAN) / STD).transpose(2, 0, 1)
    return np.expand_dims(np.ascontiguousarray(img), 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="QuantONNX path")
    parser.add_argument("--axmodel", required=True, help="compiled.axmodel path")
    parser.add_argument("--image-dir", required=True, help="directory containing input images")
    parser.add_argument("--tmp-dir", default="/tmp/compare_det_onnx_ax")
    parser.add_argument("--samples", type=int, default=0, help="0 means all images")
    parser.add_argument("--start", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs_info = [
        (o.name, o.shape, onnx_type_to_np_dtype(o.type)) for o in session.get_outputs()
    ]
    print(f"onnx input: {input_name}; outputs: {outputs_info}")

    img_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg"))
                       + glob.glob(os.path.join(args.image_dir, "*.jpeg"))
                       + glob.glob(os.path.join(args.image_dir, "*.png")))
    if not img_paths:
        raise SystemExit(f"no images under {args.image_dir}")
    img_paths = img_paths[args.start:]
    if args.samples:
        img_paths = img_paths[:args.samples]

    tmp_dir = os.path.abspath(args.tmp_dir)
    input_root = os.path.join(tmp_dir, "inputBin_det")
    output_root = os.path.join(tmp_dir, "outputBin_det")
    os.makedirs(input_root, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    results = []
    for img_path in img_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        blob = preprocess(img_path)
        onnx_outputs = session.run(None, {input_name: blob})

        input_dir = os.path.join(input_root, name)
        output_dir = os.path.join(output_root, name)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        blob.tofile(os.path.join(input_dir, f"{input_name}.bin"))

        proc = subprocess.run(
            ["pulsar2", "run", "--model", os.path.abspath(args.axmodel),
             "--input_dir", input_dir, "--output_dir", output_dir],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"{name}: pulsar2 run failed\n{proc.stdout}\n{proc.stderr}")
            continue

        for out_name, out_shape, out_dtype in outputs_info:
            ax_path = os.path.join(output_dir, f"{out_name}.bin")
            ax_raw = np.fromfile(ax_path, dtype=out_dtype)
            ax = ax_raw.reshape([d if isinstance(d, int) else blob.shape[i]
                                 for i, d in enumerate(out_shape)])
            ref = onnx_outputs[0]
            if ax.dtype != np.float32:
                # quantized integer output layer: cannot dequantize without
                # scale/zp here; compare quantized ints against round(ref)?
                # For this export the maps output is float32; guard anyway.
                print(f"{name}/{out_name}: non-float32 output {ax.dtype}, skipped")
                continue
            diff = np.abs(ax - ref)
            denom = np.linalg.norm(ax) * np.linalg.norm(ref)
            cos = float(np.dot(ax.flat, ref.flat) / denom) if denom else 0.0
            mae = float(diff.mean())
            max_abs = float(diff.max())
            results.append((name, out_name, cos, mae, max_abs))
            print(f"{name}/{out_name}: cos={cos:.6f} mae={mae:.6e} max_abs={max_abs:.6e}")

    if results:
        cos_vals = [r[2] for r in results]
        mae_vals = [r[3] for r in results]
        max_vals = [r[4] for r in results]
        print("=" * 50)
        print(f"summary over {len(results)} outputs: "
              f"cos min={min(cos_vals):.6f} mean={np.mean(cos_vals):.6f} | "
              f"mae mean={np.mean(mae_vals):.6e} | max_abs max={max(max_vals):.6e}")


if __name__ == "__main__":
    sys.exit(main())
