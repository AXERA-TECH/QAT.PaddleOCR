import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import paddle
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
PADDLEOCR_ROOT = ROOT_DIR / "references/PaddleOCR"
for source_root in (ROOT_DIR, PADDLEOCR_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from ppocr.modeling.architectures import build_model as build_paddle_model
from pytorchocr.training import build_det_model, load_ocr_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PP-OCR detection maps in Paddle and PyTorch."
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--paddle-weights", required=True)
    parser.add_argument("--torch-weights", required=True)
    parser.add_argument("--image-shape", nargs=3, type=int, default=[3, 640, 640])
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def build_paddle(config, weights_path):
    architecture = copy.deepcopy(config["Architecture"])
    model = build_paddle_model(architecture)
    model.set_state_dict(paddle.load(weights_path))
    model.eval()
    return model


def paddle_maps(output):
    if not isinstance(output, dict) or "maps" not in output:
        raise RuntimeError(
            f"Expected Paddle detection output dict with 'maps', got {type(output)!r}."
        )
    return output["maps"]


def build_torch(config_path, weights_path):
    wrapper = build_det_model(
        config_path,
        weights_path=weights_path,
        reparameterize=False,
        graph_mode="inference",
    )
    # build_det_model returns the QAT capture state, where backbone BN layers
    # intentionally remain in training mode. Float parity must use the same
    # running-statistics inference state as Paddle.
    wrapper.eval()
    return wrapper


def main(args):
    config = load_ocr_config(args.model_config)
    rng = np.random.default_rng(args.seed)
    images = rng.standard_normal([1, *args.image_shape], dtype=np.float32)
    paddle_model = build_paddle(config, args.paddle_weights)
    torch_model = build_torch(args.model_config, args.torch_weights)
    with paddle.no_grad():
        paddle_output = paddle_maps(paddle_model(paddle.to_tensor(images))).numpy()
    with torch.no_grad():
        torch_output = torch_model(torch.from_numpy(images)).cpu().numpy()
    if paddle_output.shape != torch_output.shape:
        raise RuntimeError(
            f"Output shape mismatch: Paddle {paddle_output.shape}, "
            f"PyTorch {torch_output.shape}"
        )
    difference = np.abs(paddle_output - torch_output)
    reference_abs_mean = float(np.mean(np.abs(paddle_output)))
    result = {
        "shape": list(paddle_output.shape),
        "mae": float(np.mean(difference)),
        "relative_mae": float(np.mean(difference)) / max(reference_abs_mean, 1.0e-12),
        "p99": float(np.quantile(difference, 0.99)),
        "max_abs": float(np.max(difference)),
        "reference_abs_mean": reference_abs_mean,
        "paddle_finite": bool(np.isfinite(paddle_output).all()),
        "torch_finite": bool(np.isfinite(torch_output).all()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
