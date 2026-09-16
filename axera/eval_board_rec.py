"""Board-side recognition eval: AxEngine NPU inference, self-contained.

Includes data preprocessing (equal-height resize + right zero padding +
[-1, 1] normalization, identical to the training pipeline), so no separate
prep script or pre-baked input archive is required. Images are read from the
mounted dataset and preprocessed batch by batch (low memory). A pure-PIL
fallback (decode and resize) is used when cv2 is missing on the board.

Two modes, one metric implementation:
  * online  (default): read images + labels, preprocess on the fly -- best for
    small/medium sets (e.g. ICDAR2015 test 2077 images);
  * sharded (--parts-dir, --texts): reuse preprocessed npz shards produced by
    --make-parts -- faster on slow NFS and low-memory boards.

Usage (defaults assume the mounted dataset layout):
    python3 eval_board_rec.py --axmodel M.axmodel --dictionary dict.txt
    python3 eval_board_rec.py --label-file rec_gt_test.txt --data-dir DIR \
        --make-parts PARTS --parts-size 500          # one-off shard build
    python3 eval_board_rec.py --axmodel M.axmodel --dictionary dict.txt \
        --parts-dir PARTS --texts texts.json --batch-size 1

Label image paths may be relative (to --data-dir) or absolute; absolute
paths are resolved by basename under --data-dir candidates (val/, test/, .).
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

try:
    import axengine as ort

    print("Using AXEngine (NPU)", flush=True)
except ImportError:  # pragma: no cover - board-only path
    import onnxruntime as ort

    print("Using onnxruntime (CPU)", flush=True)

PROVIDERS = ["AxEngineExecutionProvider", "CPUExecutionProvider"]

try:
    import cv2
except ImportError:  # fallback: PIL decodes and resizes (slightly different filtering)
    cv2 = None
    from PIL import Image


# ---------------------------------------------------------------------------
# Preprocessing (identical to tools/data conversion pipeline)
# ---------------------------------------------------------------------------

def resize_image(image, width, height):
    """Resize a BGR uint8 array to (width, height) with bilinear interpolation."""
    if cv2 is not None:
        return cv2.resize(image, (width, height)).astype(np.float32)
    from PIL import Image  # cv2 is unavailable: PIL is the only decoder/resizer

    resized = Image.fromarray(image[:, :, ::-1]).resize((width, height), Image.BILINEAR)
    return np.asarray(resized)[:, :, ::-1].astype(np.float32)  # RGB -> BGR


def preprocess_image(image, image_shape):
    """Equal-height resize + right zero padding + [-1, 1] normalization."""
    _, target_height, target_width = image_shape
    height, width = image.shape[:2]
    resized_width = min(target_width, int(math.ceil(target_height * width / height)))
    resized = resize_image(image, resized_width, target_height)
    resized = resized.transpose(2, 0, 1) / 255.0
    resized = (resized - 0.5) / 0.5
    output = np.zeros(image_shape, dtype=np.float32)
    output[:, :, :resized_width] = resized
    return output


def load_image(path):
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot decode image: {path}")
        return image
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))[:, :, ::-1]  # RGB -> BGR


def resolve_image(label_path, data_dir):
    """Map a label path (relative or absolute) to a file under data_dir."""
    if os.path.isabs(label_path):
        candidates = [
            os.path.join(data_dir, "val", os.path.basename(label_path)),
            os.path.join(data_dir, "test", os.path.basename(label_path)),
            os.path.join(data_dir, os.path.basename(label_path)),
            label_path,
        ]
    else:
        candidates = [os.path.join(data_dir, label_path)]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# CTC decode
# ---------------------------------------------------------------------------

def load_dictionary(path, use_space_char=True):
    characters = [l.rstrip("\r\n") for l in open(path, encoding="utf-8")]
    if use_space_char:
        characters.append(" ")  # v6 rec config: use_space_char: true
    return characters


def decode_indices(indices, characters):
    texts = []
    ignored = {0}
    class_count = len(characters) + 1  # blank(0) + characters(1..N)
    for sequence in indices:
        decoded = []
        previous = None
        for raw_index in sequence:
            index = int(raw_index)
            duplicate = previous == index
            previous = index
            if duplicate or index in ignored:
                continue
            if not 0 <= index < class_count:
                raise ValueError(f"CTC class index {index} out of range")
            decoded.append(characters[index - 1])
        texts.append("".join(decoded))
    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    default_root = "/root/heqi/project/self-developed/dataset/rec"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axmodel",
        "--onnx",
        dest="axmodel",
        default=os.path.join(default_root, "exp22a_compiled.axmodel"),
    )
    parser.add_argument(
        "--dictionary",
        "--dict",
        dest="dictionary",
        default=os.path.normpath(os.path.join(default_root, "..", "dict", "ppocrv6_dict.txt")),
    )
    parser.add_argument(
        "--use-space-char",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the space character to the dictionary (v6 rec: true; v5 rec: pass "
        "--no-use-space-char).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-shape", nargs=3, type=int, default=(3, 48, 320))
    # shard mode (preprocessed npz parts)
    parser.add_argument("--texts", default=None, help="Ground-truth JSON for --parts-dir mode")
    parser.add_argument("--parts-dir", default=None, help="Preprocessed npz shard directory")
    # online mode (self-contained preprocessing)
    parser.add_argument("--label-file", default=os.path.join(default_root, "rec_gt_test.txt"))
    parser.add_argument("--data-dir", default=default_root)
    parser.add_argument(
        "--make-parts",
        default=None,
        help="Preprocess the label set into npz shards at this directory and exit "
        "(use --parts-dir afterwards; useful for very large sets on slow NFS).",
    )
    parser.add_argument("--parts-size", type=int, default=500, help="Images per shard for --make-parts")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N samples")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    return parser.parse_args()


def make_parts(args, image_shape, label_images):
    """Preprocess the whole label set into npz shards (low memory, streaming)."""
    os.makedirs(args.make_parts, exist_ok=True)
    shard, shard_index, written = [], 0, 0
    for label_path in label_images:
        resolved = resolve_image(label_path, args.data_dir)
        if resolved is None:
            raise FileNotFoundError(f"image not found for label: {label_path}")
        shard.append(preprocess_image(load_image(resolved), image_shape))
        written += 1
        if len(shard) >= args.parts_size:
            np.savez_compressed(
                os.path.join(args.make_parts, f"part_{shard_index:04d}.npz"),
                data=np.stack(shard),
            )
            shard_index += 1
            shard = []
        if written % 5000 == 0:
            print(f"  parts: {written}/{len(label_images)}", flush=True)
    if shard:
        np.savez_compressed(
            os.path.join(args.make_parts, f"part_{shard_index:04d}.npz"),
            data=np.stack(shard),
        )
        shard_index += 1
    print(f"wrote {shard_index} shards ({written} images) -> {args.make_parts}", flush=True)


def main():
    args = parse_args()
    if args.parts_dir and args.make_parts:
        raise SystemExit("--make-parts reads images from --label-file; do not combine with --parts-dir")
    image_shape = tuple(args.image_shape)
    characters = load_dictionary(args.dictionary, args.use_space_char)

    if args.parts_dir:
        if not args.texts:
            raise SystemExit("--parts-dir requires --texts (ground-truth JSON)")
        texts = json.load(open(args.texts, encoding="utf-8"))
        label_images = None
        if args.limit:
            texts = texts[: args.limit]
    else:
        rows = [l.rstrip("\r\n") for l in open(args.label_file, encoding="utf-8") if l.strip()]
        label_images = [row.split("\t", 1)[0] for row in rows]
        texts = [row.split("\t", 1)[1].strip() for row in rows]
        if args.limit:
            texts, label_images = texts[: args.limit], label_images[: args.limit]

    if args.make_parts:
        make_parts(args, image_shape, label_images)
        return

    session = ort.InferenceSession(args.axmodel, sess_options=ort.SessionOptions(), providers=PROVIDERS)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    static_batch = input_meta.shape[0]
    if isinstance(static_batch, int) and static_batch > 0 and args.batch_size != static_batch:
        raise SystemExit(
            f"--batch-size {args.batch_size} does not match the model's static batch "
            f"{static_batch} ({input_name} {input_meta.shape}); "
            f"pass --batch-size {static_batch}"
        )
    print(
        f"model={args.axmodel} samples={len(texts)} "
        f"mode={'parts' if args.parts_dir else 'online'} providers={session.get_providers()}",
        flush=True,
    )

    preds = []
    started = time.time()
    total = len(texts)
    if args.parts_dir:
        parts = sorted(glob.glob(os.path.join(args.parts_dir, "part_*.npz")))
        print(f"parts mode: {len(parts)} shards", flush=True)
        for part in parts:
            data = np.load(part)["data"]
            for start in range(0, data.shape[0], args.batch_size):
                chunk = data[start : start + args.batch_size]
                logits = session.run(None, {input_name: chunk.astype(np.float32)})[0]
                preds.extend(decode_indices(np.argmax(logits, axis=-1), characters))
                if len(preds) >= total:
                    break
            del data
            if len(preds) >= total:
                break
        preds = preds[:total]  # --limit may cover fewer samples than the shards hold
    else:
        pending = []
        processed = 0
        for label_path in label_images:
            resolved = resolve_image(label_path, args.data_dir)
            if resolved is None:
                raise FileNotFoundError(f"image not found for label: {label_path}")
            pending.append(preprocess_image(load_image(resolved), image_shape))
            processed += 1
            if len(pending) >= args.batch_size:
                logits = session.run(None, {input_name: np.stack(pending).astype(np.float32)})[0]
                preds.extend(decode_indices(np.argmax(logits, axis=-1), characters))
                pending = []
            if processed % 5000 == 0:
                rate = processed / max(time.time() - started, 1e-6)
                print(f"  {processed}/{len(texts)} ({rate:.0f} img/s)", flush=True)
        if pending:
            logits = session.run(None, {input_name: np.stack(pending).astype(np.float32)})[0]
            preds.extend(decode_indices(np.argmax(logits, axis=-1), characters))

    total = len(texts)
    if len(preds) != total:
        raise ValueError(f"prediction count {len(preds)} != reference text count {total}")
    matches = sum(1 for pred, truth in zip(preds, texts) if pred == truth)
    import difflib

    neds = [difflib.SequenceMatcher(None, pred, truth).ratio() for pred, truth in zip(preds, texts)]
    acc = matches / total
    norm_edit_sim = sum(neds) / total
    print(f"acc={acc:.6f} norm_edit_sim={norm_edit_sim:.6f} samples={total}", flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "acc": acc,
                    "norm_edit_sim": norm_edit_sim,
                    "samples": total,
                    "axmodel": args.axmodel,
                    "label_file": args.label_file,
                    "data_dir": args.data_dir,
                    "parts_dir": args.parts_dir,
                    "seconds": time.time() - started,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
