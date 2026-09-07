import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.ao.quantization import (
    disable_fake_quant,
    disable_observer,
    move_exported_model_to_eval,
)
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pytorchocr.quantization import load_axera_quantizer
from pytorchocr.diagnostics import (
    ErrorAccumulator,
    compute_recognition_pair,
    fake_quant_state,
    new_recognition_pair,
    operator_delta,
    outputs_as_tuple,
    state_preservation,
    update_recognition_pair,
)
from pytorchocr.training import (
    build_dataset,
    build_task_model,
    build_validation_metric,
    file_sha256,
    load_ocr_config,
)
from pytorchocr.quantization import FullRecTrainingWrapper, build_qat_dynamic_shapes


STAGES = ("eager", "exported", "prepared_fake_off")
PAIRS = (
    ("eager_to_exported", "eager", "exported"),
    ("exported_to_prepared_fake_off", "exported", "prepared_fake_off"),
    ("eager_to_prepared_fake_off", "eager", "prepared_fake_off"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare recognition eager/exported/prepared PT2E outputs with "
            "observer and fake quant disabled."
        )
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--qat-config", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=0, help="0 evaluates all samples.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-shape", nargs=3, type=int, default=[3, 48, 320])
    parser.add_argument(
        "--reparameterize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rec-graph",
        choices=("deploy", "pretrained_train"),
        default="deploy",
        help="Compare the CTC deployment graph or the full CTC+NRTR training graph.",
    )
    parser.add_argument(
        "--disable-torch-tf32",
        action="store_true",
        help="Disable CUDA TF32 and record the comparison backend contract.",
    )
    return parser.parse_args()


def compare_full_training_graph(args, config):
    image_shape = tuple(args.image_shape)
    dataset = build_dataset(
        "rec",
        args.model_config,
        config,
        image_shape,
        args.label_file,
        args.data_dir,
        rec_multi_head=True,
        augmentation="none",
    )
    if args.samples:
        dataset = Subset(dataset, range(min(args.samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    example_images, example_targets = next(iter(loader))
    example_gtc_targets = example_targets["gtc_targets"]
    model = build_task_model(
        "rec",
        args.model_config,
        weights_path=args.weights,
        reparameterize=args.reparameterize,
        rec_graph="pretrained_train",
    )
    eager = FullRecTrainingWrapper(
        model,
        max_text_length=int(config["Global"].get("max_text_length", 25)),
    ).set_qat_capture_mode()
    dynamic_shapes = None
    if example_images.shape[0] > 1:
        dynamic_shapes = build_qat_dynamic_shapes(
            example_images,
            dynamic_batch=True,
            batch_aligned_inputs=1,
            max_batch=args.batch_size,
        )
    exported = torch.export.export_for_training(
        copy.deepcopy(eager),
        (example_images, example_gtc_targets),
        dynamic_shapes=dynamic_shapes,
    ).module()
    prepared = prepare_qat_pt2e(
        copy.deepcopy(exported),
        load_axera_quantizer(args.qat_config),
    )
    state_preservation_report = state_preservation(exported, prepared)
    operator_delta_report = operator_delta(exported, prepared)
    state_before_disable = fake_quant_state(prepared)
    prepared.apply(disable_observer)
    prepared.apply(disable_fake_quant)
    state_after_disable = fake_quant_state(prepared)
    if state_after_disable["modules"] == 0:
        raise RuntimeError("Prepared PT2E graph has no fake-quant modules.")
    if state_after_disable["observer_enabled"] or state_after_disable["fake_quant_enabled"]:
        raise RuntimeError("Observer or fake quant remains enabled after disable.")

    eager.set_validation_mode()
    move_exported_model_to_eval(exported)
    move_exported_model_to_eval(prepared)
    device = torch.device(args.device)
    models = {
        "eager": eager.to(device),
        "exported": exported.to(device),
        "prepared_fake_off": prepared.to(device),
    }
    metrics = {
        stage: build_validation_metric("rec", config, config_path=args.model_config)
        for stage in STAGES
    }
    output_names = FullRecTrainingWrapper.output_names
    pair_errors = {
        pair_name: {name: ErrorAccumulator() for name in output_names}
        for pair_name, _, _ in PAIRS
    }
    ctc_pairs = {pair_name: new_recognition_pair() for pair_name, _, _ in PAIRS}
    finite = {stage: {name: True for name in output_names} for stage in STAGES}
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            gtc_targets = targets["gtc_targets"].to(device, non_blocking=True)
            outputs = {
                stage: outputs_as_tuple(model(images, gtc_targets))
                for stage, model in models.items()
            }
            for stage, stage_outputs in outputs.items():
                if len(stage_outputs) != len(output_names):
                    raise RuntimeError(
                        f"{stage} returned {len(stage_outputs)} outputs; "
                        f"expected {len(output_names)}."
                    )
                for name, output in zip(output_names, stage_outputs):
                    finite[stage][name] &= bool(torch.isfinite(output).all())
                metrics[stage].update(stage_outputs[0], targets)
            for pair_name, reference_stage, actual_stage in PAIRS:
                reference_outputs = outputs[reference_stage]
                actual_outputs = outputs[actual_stage]
                for name, reference, actual in zip(
                    output_names,
                    reference_outputs,
                    actual_outputs,
                ):
                    pair_errors[pair_name][name].update(reference, actual)
                update_recognition_pair(
                    ctc_pairs[pair_name],
                    reference_outputs[0],
                    actual_outputs[0],
                )

    result = {
        "task": "rec",
        "graph_role": "pretrained_train",
        "output_names": list(output_names),
        "model_config": str(Path(args.model_config).resolve()),
        "model_config_sha256": file_sha256(args.model_config),
        "weights": str(Path(args.weights).resolve()),
        "weights_sha256": file_sha256(args.weights),
        "qat_config": str(Path(args.qat_config).resolve()),
        "qat_config_sha256": file_sha256(args.qat_config),
        "label_file": str(Path(args.label_file).resolve()),
        "label_file_sha256": file_sha256(args.label_file),
        "device": str(device),
        "torch_tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
        "sample_count": len(dataset),
        "batch_size": args.batch_size,
        "image_shape": list(image_shape),
        "reparameterized": args.reparameterize,
        "nodes": {
            "exported": len(list(exported.graph.nodes)),
            "prepared": len(list(prepared.graph.nodes)),
        },
        "state_preservation": state_preservation_report,
        "prepared_operator_delta": operator_delta_report,
        "fake_quant_state": {
            "before_disable": state_before_disable,
            "after_disable": state_after_disable,
        },
        "finite": finite,
        "metrics": {stage: metric.compute() for stage, metric in metrics.items()},
        "comparisons": {
            pair_name: {
                "outputs": {
                    name: accumulator.compute()
                    for name, accumulator in pair_errors[pair_name].items()
                },
                "ctc_behavior": compute_recognition_pair(ctc_pairs[pair_name]),
            }
            for pair_name, _, _ in PAIRS
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "graph_role": result["graph_role"],
                "sample_count": result["sample_count"],
                "nodes": result["nodes"],
                "metrics": result["metrics"],
                "comparisons": result["comparisons"],
            },
            indent=2,
        )
    )
    return result


def main(args):
    if args.samples < 0 or args.batch_size <= 0:
        raise ValueError("--samples must be non-negative and --batch-size must be positive.")
    if args.disable_torch_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260805)
    config = load_ocr_config(args.model_config)
    if args.rec_graph == "pretrained_train":
        return compare_full_training_graph(args, config)
    capture_model = build_task_model(
        "rec",
        args.model_config,
        weights_path=args.weights,
        reparameterize=args.reparameterize,
    )
    capture_batch = 2
    capture_images = torch.randn(capture_batch, *args.image_shape)
    dynamic_batch = torch.export.Dim("batch", min=1)
    exported = torch.export.export_for_training(
        capture_model,
        (capture_images,),
        dynamic_shapes=({0: dynamic_batch},),
    ).module()
    quantizer = load_axera_quantizer(args.qat_config)
    prepared = prepare_qat_pt2e(copy.deepcopy(exported), quantizer)
    state_preservation_report = state_preservation(exported, prepared)
    operator_delta_report = operator_delta(exported, prepared)
    state_before_disable = fake_quant_state(prepared)
    prepared.apply(disable_observer)
    prepared.apply(disable_fake_quant)
    state_after_disable = fake_quant_state(prepared)
    if state_after_disable["modules"] == 0:
        raise RuntimeError("Prepared PT2E graph has no fake-quant modules.")
    if state_after_disable["observer_enabled"] or state_after_disable["fake_quant_enabled"]:
        raise RuntimeError("Observer or fake quant remains enabled after disable.")

    eager = build_task_model(
        "rec",
        args.model_config,
        weights_path=args.weights,
        reparameterize=args.reparameterize,
    )
    eager.set_validation_mode()
    move_exported_model_to_eval(exported)
    move_exported_model_to_eval(prepared)
    device = torch.device(args.device)
    models = {
        "eager": eager.to(device),
        "exported": exported.to(device),
        "prepared_fake_off": prepared.to(device),
    }

    dataset = build_dataset(
        "rec",
        args.model_config,
        config,
        tuple(args.image_shape),
        args.label_file,
        args.data_dir,
    )
    if args.samples:
        dataset = Subset(dataset, range(min(args.samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    metrics = {
        stage: build_validation_metric("rec", config, config_path=args.model_config)
        for stage in STAGES
    }
    pair_stats = {name: new_recognition_pair() for name, _, _ in PAIRS}
    finite = {stage: True for stage in STAGES}
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = {stage: model(images) for stage, model in models.items()}
            for stage, output in outputs.items():
                finite[stage] &= bool(torch.isfinite(output).all())
                metrics[stage].update(output, targets)
            for name, reference_stage, actual_stage in PAIRS:
                update_recognition_pair(
                    pair_stats[name],
                    outputs[reference_stage],
                    outputs[actual_stage],
                )

    result = {
        "task": "rec",
        "model_config": str(Path(args.model_config).resolve()),
        "model_config_sha256": file_sha256(args.model_config),
        "weights": str(Path(args.weights).resolve()),
        "weights_sha256": file_sha256(args.weights),
        "qat_config": str(Path(args.qat_config).resolve()),
        "qat_config_sha256": file_sha256(args.qat_config),
        "label_file": str(Path(args.label_file).resolve()),
        "label_file_sha256": file_sha256(args.label_file),
        "device": str(device),
        "torch_tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
        "sample_count": len(dataset),
        "batch_size": args.batch_size,
        "image_shape": list(args.image_shape),
        "reparameterized": args.reparameterize,
        "nodes": {
            "exported": len(list(exported.graph.nodes)),
            "prepared": len(list(prepared.graph.nodes)),
        },
        "state_preservation": state_preservation_report,
        "prepared_operator_delta": operator_delta_report,
        "fake_quant_state": {
            "before_disable": state_before_disable,
            "after_disable": state_after_disable,
        },
        "finite": finite,
        "metrics": {stage: metric.compute() for stage, metric in metrics.items()},
        "comparisons": {
            name: compute_recognition_pair(stats)
            for name, stats in pair_stats.items()
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "sample_count": result["sample_count"],
                "nodes": result["nodes"],
                "state_preservation": result["state_preservation"],
                "fake_quant_state": {
                    name: {
                        key: value
                        for key, value in state.items()
                        if key != "module_states"
                    }
                    for name, state in result["fake_quant_state"].items()
                },
                "metrics": result["metrics"],
                "comparisons": result["comparisons"],
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    main(parse_args())
