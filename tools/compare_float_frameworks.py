import argparse
import copy
import json
import logging
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import paddle
import torch
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
PADDLEOCR_ROOT = ROOT_DIR / "references/PaddleOCR"
PYTORCHOCR_ROOT = ROOT_DIR / "references/PytorchOCR"
for source_root in (ROOT_DIR, PADDLEOCR_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from ppocr.data import build_dataloader
from ppocr.metrics import build_metric
from ppocr.modeling.architectures import build_model as build_paddle_model
from ppocr.postprocess import build_post_process
from pytorchocr.training import (
    build_det_model,
    build_rec_model,
    load_ocr_config,
    rec_out_channels,
)
from pytorchocr.utils.hashing import file_sha256


class DifferenceAccumulator:
    def __init__(self, samples_per_update=4096):
        self.count = 0
        self.sum_abs = 0.0
        self.max_abs = 0.0
        self.samples = []
        self.samples_per_update = samples_per_update
        self.worst_sample_index = None
        self.worst_sample_max_abs = 0.0

    def update(self, reference, candidate, sample_offset=0):
        reference = np.asarray(reference)
        candidate = np.asarray(candidate)
        if reference.shape != candidate.shape:
            raise RuntimeError(
                f"Tensor shape mismatch: {reference.shape} != {candidate.shape}"
            )
        difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        self.count += difference.size
        self.sum_abs += float(difference.sum())
        self.max_abs = max(self.max_abs, float(difference.max(initial=0.0)))
        if difference.ndim:
            per_sample = difference.reshape(difference.shape[0], -1).max(axis=1)
            local_index = int(np.argmax(per_sample))
            if float(per_sample[local_index]) > self.worst_sample_max_abs:
                self.worst_sample_max_abs = float(per_sample[local_index])
                self.worst_sample_index = sample_offset + local_index
        flat = difference.reshape(-1)
        if flat.size <= self.samples_per_update:
            sample = flat
        else:
            indices = np.linspace(
                0, flat.size - 1, self.samples_per_update, dtype=np.int64
            )
            sample = flat[indices]
        self.samples.append(sample)

    def result(self):
        values = np.concatenate(self.samples) if self.samples else np.empty(0)
        return {
            "count": self.count,
            "mae": self.sum_abs / self.count if self.count else 0.0,
            "p99": float(np.quantile(values, 0.99)) if values.size else 0.0,
            "p99_sample_count": int(values.size),
            "max_abs": self.max_abs,
            "worst_sample_index": self.worst_sample_index,
            "worst_sample_max_abs": self.worst_sample_max_abs,
        }


class BehaviorAccumulator:
    def __init__(self, task, coordinate_tolerance=1.0):
        self.task = task
        self.coordinate_tolerance = coordinate_tolerance
        self.sample_count = 0
        self.match_count = 0
        self.first_mismatch_index = None
        self.box_count_match = 0
        self.coordinate_comparable = 0
        self.coordinate_match = 0
        self.max_coordinate_abs = 0.0
        self.mismatches = []

    def update(self, reference, candidate, sample_offset):
        if len(reference) != len(candidate):
            raise RuntimeError("Postprocess batch size mismatch.")
        for local_index, (expected, actual) in enumerate(zip(reference, candidate)):
            sample_index = sample_offset + local_index
            if self.task == "rec":
                matches = expected[0] == actual[0]
            else:
                expected_points = np.asarray(expected["points"])
                actual_points = np.asarray(actual["points"])
                count_matches = len(expected_points) == len(actual_points)
                self.box_count_match += int(count_matches)
                matches = False
                if count_matches and expected_points.shape == actual_points.shape:
                    self.coordinate_comparable += 1
                    coordinate_abs = float(
                        np.max(np.abs(expected_points - actual_points), initial=0.0)
                    )
                    self.max_coordinate_abs = max(
                        self.max_coordinate_abs, coordinate_abs
                    )
                    coordinate_matches = coordinate_abs <= self.coordinate_tolerance
                    self.coordinate_match += int(coordinate_matches)
                    matches = coordinate_matches
            self.sample_count += 1
            self.match_count += int(matches)
            if not matches and self.first_mismatch_index is None:
                self.first_mismatch_index = sample_index
            if not matches and len(self.mismatches) < 64:
                mismatch = {"sample_index": sample_index}
                if self.task == "rec":
                    mismatch.update(
                        {
                            "paddle_text": expected[0],
                            "candidate_text": actual[0],
                            "paddle_confidence": float(expected[1]),
                            "candidate_confidence": float(actual[1]),
                        }
                    )
                else:
                    mismatch.update(
                        {
                            "paddle_box_count": len(expected_points),
                            "candidate_box_count": len(actual_points),
                        }
                    )
                self.mismatches.append(mismatch)

    def result(self):
        result = {
            "sample_count": self.sample_count,
            "agreement": self.match_count / self.sample_count,
            "first_mismatch_index": self.first_mismatch_index,
            "mismatches": self.mismatches,
        }
        if self.task == "det":
            result.update(
                {
                    "box_count_agreement": self.box_count_match / self.sample_count,
                    "coordinate_comparable_samples": self.coordinate_comparable,
                    "coordinate_agreement_atol_1px": (
                        self.coordinate_match / self.coordinate_comparable
                        if self.coordinate_comparable
                        else 0.0
                    ),
                    "max_coordinate_abs": self.max_coordinate_abs,
                }
            )
        return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Paddle, PytorchOCR, and project float models on one Paddle eval stream."
    )
    parser.add_argument("task", choices=("det", "rec"))
    parser.add_argument("--paddle-config", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--paddle-weights", required=True)
    parser.add_argument("--torch-weights", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--pytorchocr-root", default=str(PYTORCHOCR_ROOT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--reparameterize-route2",
        action="store_true",
        help="Fold route2 Conv/BN reparameterization before float comparison.",
    )
    return parser.parse_args()


def load_yaml(path):
    with open(path, "rb") as stream:
        return yaml.safe_load(stream)


def git_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT_DIR,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def configure_eval(config, args):
    config = copy.deepcopy(config)
    config["Eval"]["dataset"]["data_dir"] = str(Path(args.data_dir).resolve())
    config["Eval"]["dataset"]["label_file_list"] = [
        str(Path(args.label_file).resolve())
    ]
    config["Eval"]["loader"]["num_workers"] = args.workers
    if args.batch_size is not None:
        config["Eval"]["loader"]["batch_size_per_card"] = args.batch_size
    config["Global"]["character_dict_path"] = str(
        Path(args.route_config).resolve().parents[3]
        / "pytorchocr/utils/dict/ppocrv5_dict.txt"
    ) if args.task == "rec" else config["Global"].get("character_dict_path")
    return config


def build_paddle(config, route_config, weights_path, task):
    architecture = copy.deepcopy(config["Architecture"])
    if task == "rec":
        architecture["Head"]["out_channels_list"] = rec_out_channels(
            load_ocr_config(route_config), config_path=route_config
        )
    model = build_paddle_model(architecture)
    model.set_state_dict(paddle.load(weights_path))
    model.eval()
    if task == "rec":
        model.head.ctc_head.train()
    return model


def build_route2(route_config, weights_path, task, device, reparameterize=False):
    if task == "det":
        model = build_det_model(
            route_config,
            weights_path=weights_path,
            reparameterize=reparameterize,
            graph_mode="inference",
        )
        model.eval()
    else:
        model = build_rec_model(
            route_config,
            weights_path=weights_path,
            reparameterize=reparameterize,
        )
        model.model.eval()
        model.model.head.ctc_head.train()
    return model.to(device)


def import_pytorchocr_modeling(project_root):
    package_root = Path(project_root).resolve() / "torchocr"
    if not package_root.is_dir():
        raise FileNotFoundError(f"Missing PytorchOCR package: {package_root}")
    package = types.ModuleType("torchocr")
    package.__path__ = [str(package_root)]
    package.__package__ = "torchocr"
    sys.modules["torchocr"] = package
    from torchocr.modeling.architectures import build_model

    return build_model


def build_pytorchocr(project_root, route_config, weights_path, task, device):
    build_model = import_pytorchocr_modeling(project_root)
    config = load_ocr_config(route_config)
    architecture = copy.deepcopy(config["Architecture"])
    if task == "rec":
        architecture["Head"]["out_channels_list"] = rec_out_channels(
            config, config_path=route_config
        )
    model = build_model(architecture)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if task == "det":
        model.load_state_dict(state, strict=True)
        load_status = "strict_all_parameters"
    else:
        ctc_state = {
            name: value
            for name, value in state.items()
            if not name.startswith(("head.before_gtc.", "head.gtc_head."))
        }
        incompatible = model.load_state_dict(ctc_state, strict=False)
        invalid_missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith(("head.before_gtc.", "head.gtc_head."))
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "PytorchOCR CTC weights are incompatible: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        load_status = "strict_ctc_path_gtc_excluded"
    model.eval()
    if task == "rec":
        model.head.ctc_head.train()
    return model.to(device), load_status


def paddle_output(model, images, task):
    output = model(images)
    if task == "det":
        output = output["maps"]
    return output.numpy()


def pytorchocr_output(model, images, task):
    output = model(images)
    if task == "det":
        output = output.get("maps", output.get("res"))
        if output is None:
            raise RuntimeError("PytorchOCR detection output has no maps/res tensor.")
    elif isinstance(output, dict):
        output = output["res"]
    return output.detach().cpu().numpy()


def centered(logits):
    return logits - logits.mean(axis=-1, keepdims=True)


def softmax(logits):
    logits = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(logits)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def numpy_batch(batch):
    return [item.numpy() if isinstance(item, paddle.Tensor) else item for item in batch]


def task_components(config):
    postprocess = [
        build_post_process(config["PostProcess"], config["Global"])
        for _ in range(3)
    ]
    metrics = [build_metric(config["Metric"]) for _ in range(3)]
    return postprocess, metrics


def evaluate_task(
    task, outputs, batch, postprocesses, metrics, behavior, sample_offset
):
    results = []
    for output, postprocess, metric in zip(outputs, postprocesses, metrics):
        if task == "det":
            result = postprocess({"maps": output}, batch[1])
            metric(result, batch)
        else:
            result = postprocess(output, batch[1])
            metric(result, batch)
        results.append(result[0] if task == "rec" else result)
    for name, candidate in zip(("pytorchocr", "route2"), results[1:]):
        behavior[name].update(results[0], candidate, sample_offset)


def main(args):
    started = time.time()
    paddle_config = configure_eval(load_yaml(args.paddle_config), args)
    paddle_device = args.device.replace("cuda", "gpu")
    paddle.set_device(paddle_device)
    torch_device = torch.device(args.device)

    logger = logging.getLogger("float-framework-parity")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    loader = build_dataloader(paddle_config, "Eval", paddle_device, logger, seed=0)
    paddle_model = build_paddle(
        paddle_config, args.route_config, args.paddle_weights, args.task
    )
    route_model = build_route2(
        args.route_config,
        args.torch_weights,
        args.task,
        torch_device,
        reparameterize=args.reparameterize_route2,
    )
    reference_model, reference_load = build_pytorchocr(
        args.pytorchocr_root,
        args.route_config,
        args.torch_weights,
        args.task,
        torch_device,
    )

    names = ("pytorchocr", "route2")
    raw_stats = {name: DifferenceAccumulator() for name in names}
    centered_stats = {name: DifferenceAccumulator() for name in names}
    probability_stats = {name: DifferenceAccumulator() for name in names}
    argmax_equal = {name: 0 for name in names}
    argmax_count = {name: 0 for name in names}
    behavior = {name: BehaviorAccumulator(args.task) for name in names}
    postprocesses, metrics = task_components(paddle_config)
    sample_count = 0
    next_progress = 500

    with paddle.no_grad(), torch.no_grad():
        for batch in loader:
            images_np = batch[0].numpy()
            remaining = None if args.max_samples is None else args.max_samples - sample_count
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None and images_np.shape[0] > remaining:
                images_np = images_np[:remaining]
                batch = [item[:remaining] for item in batch]
            paddle_result = paddle_output(paddle_model, batch[0], args.task)
            torch_images = torch.from_numpy(images_np).to(torch_device)
            reference_result = pytorchocr_output(reference_model, torch_images, args.task)
            route_result = route_model(torch_images).detach().cpu().numpy()
            outputs = (paddle_result, reference_result, route_result)
            for name, candidate in zip(names, outputs[1:]):
                raw_stats[name].update(paddle_result, candidate, sample_count)
                if args.task == "rec":
                    centered_stats[name].update(
                        centered(paddle_result), centered(candidate), sample_count
                    )
                    probability_stats[name].update(
                        softmax(paddle_result), softmax(candidate), sample_count
                    )
                    paddle_argmax = np.argmax(paddle_result, axis=-1)
                    candidate_argmax = np.argmax(candidate, axis=-1)
                    argmax_equal[name] += int(np.count_nonzero(paddle_argmax == candidate_argmax))
                    argmax_count[name] += paddle_argmax.size
            batch_np = numpy_batch(batch)
            evaluate_task(
                args.task,
                outputs,
                batch_np,
                postprocesses,
                metrics,
                behavior,
                sample_count,
            )
            sample_count += images_np.shape[0]
            if sample_count >= next_progress:
                logger.info("Compared %d samples", sample_count)
                next_progress += 500

    metric_results = {
        name: metric.get_metric()
        for name, metric in zip(("paddle", "pytorchocr", "route2"), metrics)
    }
    report = {
        "task": args.task,
        "sample_count": sample_count,
        "input_source": "PaddleOCR official Eval dataloader",
        "postprocess_source": "PaddleOCR official postprocess",
        "metric_source": "PaddleOCR official metric",
        "device": args.device,
        "batch_size": paddle_config["Eval"]["loader"]["batch_size_per_card"],
        "workers": args.workers,
        "route2_reparameterized": args.reparameterize_route2,
        "pytorchocr_load_status": reference_load,
        "command": sys.argv,
        "environment": {
            "python": platform.python_version(),
            "paddle": paddle.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "git": git_state(),
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
            for name, path in {
                "paddle_config": args.paddle_config,
                "route_config": args.route_config,
                "paddle_weights": args.paddle_weights,
                "torch_weights": args.torch_weights,
                "label_file": args.label_file,
            }.items()
        },
        "tensor_difference_vs_paddle": {
            name: {"raw": raw_stats[name].result()} for name in names
        },
        "behavior_vs_paddle": {name: behavior[name].result() for name in names},
        "task_metrics": metric_results,
        "elapsed_seconds": time.time() - started,
    }
    if args.task == "rec":
        for name in names:
            report["tensor_difference_vs_paddle"][name].update(
                {
                    "centered_logits": centered_stats[name].result(),
                    "probability": probability_stats[name].result(),
                    "argmax_agreement": argmax_equal[name] / argmax_count[name],
                }
            )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
