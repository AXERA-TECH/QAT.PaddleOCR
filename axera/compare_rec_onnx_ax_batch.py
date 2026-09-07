#!/usr/bin/env python3
"""Batch compare PP-OCRv5 recognition CTC logits: QuantONNX (ORT CPU)
vs compiled.axmodel (single pulsar2 run --list session), sample by sample.

Recognition preprocessing uses equal-height resize, right zero padding,
[-1, 1] normalization, and BGR input; CTC decoding removes blanks and repeats.

Inputs written to <tmp>/inputBin/<case>/images.bin, one case per sample;
pulsar2 run --list lists all cases in one model session to avoid the
per-invocation simulator startup cost.

Usage:
    python3 compare_rec_onnx_ax_batch.py \
        --onnx   onnx/exp13_u8s8_reparam/ppocrv5_mobile_rec_exp13_u8s8_reparam_qdq.onnx \
        --axmodel output/exp13_u8s8_reparam/compiled.axmodel \
        --dictionary-path ppocrv5_dict.txt \
        --label-file /path/to/rec_gt_test.txt \
        --data-dir  /path/to/icdr2015val \
        --samples 100 --tmp-dir /tmp/rec_cmp_batch
"""

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


def load_dictionary(dictionary_path, use_space_char=False):
    with open(dictionary_path, encoding="utf-8") as stream:
        characters = [line.rstrip("\r\n") for line in stream]
    if use_space_char:
        characters.append(" ")
    return characters


class CTCEncoder:
    def __init__(self, characters, max_text_length=25):
        self.characters = list(characters)
        self.dictionary = {
            character: index + 1 for index, character in enumerate(self.characters)
        }
        self.max_text_length = int(max_text_length)

    def encode(self, text):
        if not text or len(text) > self.max_text_length:
            return None
        encoded = [self.dictionary[char] for char in text if char in self.dictionary]
        if not encoded:
            return None
        padded = encoded + [0] * (self.max_text_length - len(encoded))
        return np.asarray(padded, dtype=np.int64)


def decode_indices(indices, characters, remove_duplicates=True):
    texts = []
    ignored = {0}
    class_count = len(characters) + 1
    for sequence in indices:
        decoded = []
        previous = None
        for raw_index in sequence:
            index = int(raw_index)
            duplicate = remove_duplicates and previous == index
            previous = index
            if duplicate or index in ignored:
                continue
            if not 0 <= index < class_count:
                raise ValueError(f"CTC class index {index} out of range")
            decoded.append(characters[index - 1])
        texts.append("".join(decoded))
    return texts


def resize_rec_image(image, image_shape):
    channels, target_height, target_width = image_shape
    if channels != 3:
        raise ValueError("Recognition images must have three channels.")
    height, width = image.shape[:2]
    resized_width = min(target_width, int(math.ceil(target_height * width / height)))
    resized = cv2.resize(image, (resized_width, target_height)).astype(np.float32)
    resized = resized.transpose(2, 0, 1) / 255.0
    resized = (resized - 0.5) / 0.5
    output = np.zeros(image_shape, dtype=np.float32)
    output[:, :, :resized_width] = resized
    return output


def levenshtein_distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def logit_metrics(ax_logits, onnx_logits):
    ax = ax_logits.astype(np.float32)
    onnx = onnx_logits.astype(np.float32)
    diff = ax - onnx
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "argmax_agree": float(np.mean(ax.argmax(-1) == onnx.argmax(-1))),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--axmodel", required=True, type=Path)
    parser.add_argument("--dictionary-path", required=True, type=Path)
    parser.add_argument("--label-file", required=True, type=Path)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--image-shape", nargs=3, type=int, default=(3, 48, 320))
    parser.add_argument("--max-text-length", type=int, default=25)
    parser.add_argument("--use-space-char", action="store_true")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--tmp-dir", default="/tmp/rec_cmp_batch", type=Path)
    parser.add_argument("--keep-inputs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    image_shape = tuple(args.image_shape)
    characters = load_dictionary(args.dictionary_path, args.use_space_char)
    encoder = CTCEncoder(characters, args.max_text_length)

    with open(args.label_file, encoding="utf-8") as stream:
        raw_samples = [line.rstrip("\r\n") for line in stream if line.strip()]
    samples = []
    for line in raw_samples:
        image_name, text = line.split("\t", 1)
        encoded = encoder.encode(text)
        if encoded is not None:
            samples.append((image_name, text, encoded))
    samples = samples[args.start: args.start + args.samples]
    print(f"loaded {len(samples)} samples (start={args.start})")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(args.onnx.resolve()), sess_options=so, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"onnx input: {input_name} {session.get_inputs()[0].shape}")
    print(f"onnx output: {output_name}")

    tmp = Path(args.tmp_dir)
    input_dir_root = tmp / "inputBin"
    output_dir_root = tmp / "outputBin"
    if args.keep_inputs:
        input_dir_root.mkdir(parents=True, exist_ok=True)
    else:
        for d in (input_dir_root, output_dir_root):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

    onnx_logits = []
    ax_logits = []
    ref_texts = []
    onnx_texts = []
    case_names = []

    for index, (image_name, text, _encoded) in enumerate(tqdm(samples, desc="Prep/ORT")):
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = Path(args.data_dir) / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not decode image: {image_path}")
        tensor = resize_rec_image(image, image_shape)
        case = f"case_{index + args.start:05d}"
        case_names.append(case)
        (input_dir_root / case).mkdir(parents=True, exist_ok=True)
        tensor.reshape(-1).astype(np.float32).tofile(
            str(input_dir_root / case / f"{input_name}.bin")
        )
        onnx_logits.append(session.run(None, {input_name: tensor[None, ...]})[0][0])
        onnx_texts.append(decode_indices(onnx_logits[-1][None].argmax(axis=2), characters)[0])
        ref_texts.append(text)

    list_file = tmp / "cases.list"
    list_file.write_text("\n".join(case_names) + "\n")

    print(f"running pulsar2 run --list over {len(case_names)} cases ...")
    start = time.perf_counter()
    result = subprocess.run(
        ["pulsar2", "run", "--model", str(args.axmodel.resolve()),
         "--input_dir", str(input_dir_root), "--output_dir", str(output_dir_root),
         "--list", str(list_file)],
        capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - start
    print(f"pulsar2 run finished in {elapsed:.1f}s, returncode={result.returncode}")
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(1)

    metrics_list = []
    ax_texts = []
    for index, case in enumerate(case_names):
        out_dir = output_dir_root / case
        out_file = out_dir / f"{output_name}.bin"
        if not out_file.is_file():
            bins = sorted(out_dir.glob("*.bin"))
            if not bins:
                raise FileNotFoundError(f"no output bin in {out_dir}")
            out_file = bins[0]
        ax = np.fromfile(out_file, dtype=np.float32).reshape(onnx_logits[index].shape)
        ax_logits.append(ax)
        m = logit_metrics(ax, onnx_logits[index])
        m["index"] = index + args.start
        m["image"] = samples[index][0]
        metrics_list.append(m)
        ax_texts.append(decode_indices(ax[None].argmax(axis=2), characters)[0])

    agg = {
        "samples": len(metrics_list),
        "pulsar2_elapsed_seconds": round(elapsed, 1),
        "max_abs_overall": round(float(np.max([m["max_abs"] for m in metrics_list])), 4),
        "mean_max_abs": round(float(np.mean([m["max_abs"] for m in metrics_list])), 4),
        "mean_mae": round(float(np.mean([m["mae"] for m in metrics_list])), 5),
        "mean_rmse": round(float(np.mean([m["rmse"] for m in metrics_list])), 5),
        "mean_argmax_agree": round(float(np.mean([m["argmax_agree"] for m in metrics_list])), 5),
    }
    norm = lambda t: t.replace(" ", "")
    agg["ax_seq_acc"] = round(sum(int(norm(a) == norm(r)) for a, r in zip(ax_texts, ref_texts)) / len(ax_texts), 4)
    agg["onnx_seq_acc"] = round(sum(int(norm(o) == norm(r)) for o, r in zip(onnx_texts, ref_texts)) / len(onnx_texts), 4)
    agg["ax_vs_onnx_seq_acc"] = round(sum(int(norm(a) == norm(o)) for a, o in zip(ax_texts, onnx_texts)) / len(ax_texts), 4)

    worst = sorted(metrics_list, key=lambda m: m["max_abs"], reverse=True)[:8]
    print(json.dumps(agg, indent=2, sort_keys=True))
    print("=== worst samples (by max_abs) ===")
    for m in worst:
        print(
            f"  {m['index']:5d} {m['image']} max_abs={m['max_abs']:.3f} "
            f"mae={m['mae']:.4f} argmax_agree={m['argmax_agree']:.3f}"
        )
    with open(tmp / "per_sample.json", "w") as stream:
        json.dump(metrics_list, stream, indent=1)


if __name__ == "__main__":
    main()
