import argparse
import dataclasses
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pytorchocr.quantization import (
    FullDetTrainingWrapper,
    FullRecTrainingWrapper,
    build_qat_dynamic_shapes,
    convert_prepared_model,
    initialize_weight_observers,
    load_axera_quantizer,
    numpy_error_stats,
    qdq_stats,
    prepare_qat_model,
    run_onnx_reference,
    run_ort,
    sequence_error_stats,
    validate_qdq_graph,
)
from pytorchocr.diagnostics import (
    build_prepared_qat_checkpoint,
    prepared_random_stage_outputs,
    random_stage_comparison,
)
from pytorchocr.training import (
    build_task_model,
    load_ocr_config,
    relocate_project_path,
)
from pytorchocr.quantization.onnx_export import (
    RecTrainingDeploymentProjection,
    export_onnx,
    onnx_activation_qparams,
    prepared_activation_qparams,
    require_default_qparams,
)


@dataclass(frozen=True)
class ModelSpec:
    task: str
    model_config: str
    weights: str
    qat_config: str
    image_shape: tuple[int, int, int]


MODEL_SPECS = {
    "ppocrv5_mobile_rec": ModelSpec(
        task="rec",
        model_config="configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml",
        weights="weights/ptocr_v5_mobile_rec_full.pth",
        qat_config="configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json",
        image_shape=(3, 48, 320),
    ),
    "ppocrv5_mobile_det": ModelSpec(
        task="det",
        model_config="configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml",
        weights="weights/ptocr_v5_mobile_det.pth",
        qat_config="configs/qat/ppocrv5_mobile_det_u8s8.json",
        image_shape=(3, 640, 640),
    ),
    "ppocrv6_small_rec": ModelSpec(
        task="rec",
        model_config="configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml",
        weights="weights/ptocr_v6_small_rec_full.pth",
        qat_config="configs/qat/ppocrv6_small_rec_u8s8.json",
        image_shape=(3, 48, 320),
    ),
    "ppocrv6_small_det": ModelSpec(
        task="det",
        model_config="configs/det/PP-OCRv6/PP-OCRv6_small_det.yml",
        weights="weights/ptocr_v6_small_det_full.pth",
        qat_config="configs/qat/ppocrv6_small_det_u8s8.json",
        image_shape=(3, 640, 640),
    ),
}


def add_optimize_argument(parser):
    parser.add_argument(
        "--onnx-optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ONNXProgram.optimize() before saving (enabled by default).",
    )


def add_model_selection_arguments(parser):
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )


def add_training_arguments(parser):
    add_model_selection_arguments(parser)
    parser.add_argument(
        "--format",
        choices=("float", "quantonnx", "both"),
        default="both",
    )
    parser.add_argument("--output-dir", default="exports/training")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--float-reparameterize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reparameterize Float ONNX backbone/neck (disabled by default).",
    )
    parser.add_argument(
        "--quant-reparameterize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reparameterize before PT2E QAT capture (enabled by default).",
    )
    parser.add_argument(
        "--quant-graph",
        choices=("training", "inference"),
        default="training",
        help=(
            "QuantONNX graph role: preserve all pretrained-training outputs "
            "or export the deployment-only DB shrink/CTC path."
        ),
    )
    add_optimize_argument(parser)
    parser.add_argument(
        "--ort-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def add_float_matrix_arguments(parser):
    add_model_selection_arguments(parser)
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=("training", "inference"),
        default=("training", "inference"),
        help="Float graph roles to export (both by default).",
    )
    parser.add_argument(
        "--reparameterizations",
        nargs="+",
        choices=("reparameterized", "non_reparameterized"),
        default=("reparameterized", "non_reparameterized"),
        help="Float model structures to export (both by default).",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/float_onnx/v5_v6_reparameterization_matrix",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    add_optimize_argument(parser)
    parser.add_argument(
        "--ort-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def add_quant_matrix_arguments(parser):
    add_model_selection_arguments(parser)
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=("training", "inference"),
        default=("training", "inference"),
        help="QuantONNX graph roles to export (both by default).",
    )
    parser.add_argument(
        "--reparameterizations",
        nargs="+",
        choices=("reparameterized", "non_reparameterized"),
        default=("reparameterized", "non_reparameterized"),
        help="QuantONNX structures to export (both by default).",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/quantonnx/v5_v6_u16s16_reparameterization_matrix",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    add_optimize_argument(parser)
    parser.add_argument(
        "--ort-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def add_initialized_arguments(parser):
    parser.add_argument("--task", choices=("det", "rec"), required=True)
    parser.add_argument(
        "--det-graph",
        choices=("inference", "training"),
        default="training",
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--qat-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--image-shape", nargs=3, type=int, required=True)
    parser.add_argument(
        "--dynamic-heights",
        nargs="+",
        type=int,
        help="Allowed recognition heights for the prepared PT2E training graph only.",
    )
    parser.add_argument(
        "--reparameterize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    add_optimize_argument(parser)


def add_checkpoint_arguments(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=("det", "rec"))
    parser.add_argument("--model-config")
    parser.add_argument("--qat-config")
    parser.add_argument("--weights")
    parser.add_argument("--image-shape", nargs=3, type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    add_optimize_argument(parser)
    parser.add_argument(
        "--onnx-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the slow ONNX Python ReferenceEvaluator.",
    )
    parser.add_argument(
        "--ort-optimizer-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run ORT graph-optimization A/B as a separate diagnostic.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the batch policy recorded in checkpoint metadata.",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export PP-OCR Float ONNX and Axera PT2E QuantONNX."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    checkpoint = commands.add_parser(
        "checkpoint",
        help="Export a trained PT2E QAT checkpoint.",
    )
    add_checkpoint_arguments(checkpoint)
    checkpoint.set_defaults(handler=run_checkpoint_export)

    initialized = commands.add_parser(
        "initialized",
        help="Export a deployment graph with initialized observers.",
    )
    add_initialized_arguments(initialized)
    initialized.set_defaults(handler=run_initialized_export)

    training = commands.add_parser(
        "training",
        help="Export pretrained Float ONNX and training/inference QuantONNX.",
    )
    add_training_arguments(training)
    training.set_defaults(handler=run_training_export)

    float_matrix = commands.add_parser(
        "float-matrix",
        help=(
            "Export training/inference Float ONNX with and without "
            "deployment reparameterization."
        ),
    )
    add_float_matrix_arguments(float_matrix)
    float_matrix.set_defaults(handler=run_float_matrix_export)

    quant_matrix = commands.add_parser(
        "quant-matrix",
        help=(
            "Export v5/v6 det/rec U16 activation + S16 weight QuantONNX "
            "with and without reparameterization."
        ),
    )
    add_quant_matrix_arguments(quant_matrix)
    quant_matrix.set_defaults(handler=run_quant_matrix_export)

    audit = commands.add_parser(
        "audit",
        help="Check existing training/inference export artifacts.",
    )
    add_model_selection_arguments(audit)
    audit.add_argument("--output-dir", default="exports/training")
    audit.set_defaults(handler=run_training_audit)
    return parser


def build_training_wrapper(name, spec, reparameterize):
    if spec.task == "det":
        model = build_task_model(
            "det",
            spec.model_config,
            weights_path=spec.weights,
            reparameterize=reparameterize,
            det_graph="pretrained_train",
        )
        return FullDetTrainingWrapper(model)

    config = load_ocr_config(spec.model_config)
    model = build_task_model(
        "rec",
        spec.model_config,
        weights_path=spec.weights,
        reparameterize=reparameterize,
        rec_graph="pretrained_train",
    )
    return FullRecTrainingWrapper(
        model,
        max_text_length=config["Global"]["max_text_length"],
    )


def build_quant_export_model(name, spec, reparameterize, graph_mode):
    if graph_mode == "training":
        wrapper = build_training_wrapper(name, spec, reparameterize)
        wrapper.set_qat_capture_mode()
        return wrapper, wrapper.output_names, getattr(
            wrapper,
            "max_text_length",
            None,
        )
    if graph_mode != "inference":
        raise ValueError(f"Unsupported QuantONNX graph mode: {graph_mode}")

    model = build_task_model(
        spec.task,
        spec.model_config,
        weights_path=spec.weights,
        reparameterize=reparameterize,
        det_graph="inference",
        rec_graph="deploy",
    )
    output_names = ("maps",) if spec.task == "det" else ("logits",)
    return model, output_names, None


def build_float_export_model(name, spec, reparameterize, graph_mode):
    if graph_mode == "training":
        model = build_training_wrapper(name, spec, reparameterize)
        model.set_validation_mode()
        return model, model.output_names, getattr(model, "max_text_length", None)
    if graph_mode != "inference":
        raise ValueError(f"Unsupported Float ONNX graph mode: {graph_mode}")

    model = build_task_model(
        spec.task,
        spec.model_config,
        weights_path=spec.weights,
        reparameterize=reparameterize,
        det_graph="inference",
        rec_graph="deploy",
    )
    set_validation_mode = getattr(model, "set_validation_mode", None)
    if callable(set_validation_mode):
        set_validation_mode()
    else:
        model.eval()
    output_names = ("maps",) if spec.task == "det" else ("logits",)
    return model, output_names, None


def make_example_inputs(
    spec,
    batch_size,
    max_text_length=None,
    include_rec_targets=True,
):
    generator = torch.Generator().manual_seed(20260806)
    images = torch.randn(
        [batch_size, *spec.image_shape],
        generator=generator,
        dtype=torch.float32,
    )
    if spec.task == "det" or not include_rec_targets:
        return (images,), ("images",)

    gtc_targets = torch.zeros(
        [batch_size, max_text_length],
        dtype=torch.int64,
    )
    gtc_targets[:, 0] = 2
    gtc_targets[:, 1] = 4
    gtc_targets[:, 2] = 3
    return (images, gtc_targets), ("images", "gtc_targets")


def tensor_outputs(outputs):
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    return tuple(outputs)


def selected_outputs(outputs, output_index=None):
    outputs = tensor_outputs(outputs)
    if output_index is None:
        return outputs
    return (outputs[output_index],)


def graph_shapes(value_infos):
    result = {}
    for value in value_infos:
        shape = []
        for dimension in value.type.tensor_type.shape.dim:
            shape.append(
                dimension.dim_param if dimension.dim_param else dimension.dim_value
            )
        result[value.name] = shape
    return result


def artifact_metadata(path):
    path = Path(path)
    return {
        "file_size": path.stat().st_size,
    }


def run_ort_outputs(path, example_inputs):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if len(session.get_inputs()) != len(example_inputs):
        raise RuntimeError(
            "ONNX input count does not match the training export contract: "
            f"{len(session.get_inputs())} != {len(example_inputs)}"
        )
    feed = {
        value.name: tensor.detach().cpu().numpy()
        for value, tensor in zip(session.get_inputs(), example_inputs)
    }
    return session.run(None, feed)


def output_errors(reference, actual):
    if len(reference) != len(actual):
        raise RuntimeError(f"Output count mismatch: {len(reference)} != {len(actual)}")
    records = []
    for expected, observed in zip(reference, actual):
        expected_array = expected.detach().cpu().numpy()
        difference = np.abs(expected_array - observed)
        records.append(
            {
                "mae": float(np.mean(difference)),
                "max_abs": float(np.max(difference)),
                "finite": bool(np.isfinite(observed).all()),
            }
        )
    return records


def validate_regional_targets(prepared, qat_config_path):
    config = json.loads(Path(qat_config_path).read_text())
    graph_nodes = {node.name for node in prepared.graph.nodes}
    expected = []
    for regional in config.get("regional_configs", []):
        expected.extend(regional.get("module_names") or [])
    missing = sorted(set(expected) - graph_nodes)
    if missing:
        raise RuntimeError(
            "QAT regional targets did not match the prepared graph: "
            + ", ".join(missing)
        )
    return {
        "expected": len(set(expected)),
        "matched": len(set(expected)),
        "targets": sorted(set(expected)),
    }


def export_float(name, spec, args, output_root):
    wrapper = build_training_wrapper(name, spec, args.float_reparameterize)
    wrapper.set_validation_mode()
    max_text_length = getattr(wrapper, "max_text_length", None)
    inputs, input_names = make_example_inputs(
        spec,
        args.batch_size,
        max_text_length=max_text_length,
    )
    with torch.no_grad():
        reference = tensor_outputs(wrapper(*inputs))
    output_path = output_root / "float" / f"{name}_training_float.onnx"
    model = export_onnx(
        wrapper,
        inputs,
        output_path,
        wrapper.output_names,
        input_names=input_names,
        optimize=args.onnx_optimize,
    )
    report = {
        "format": "float",
        "reparameterized": args.float_reparameterize,
        "output": str(output_path.resolve()),
        "inputs": graph_shapes(model.graph.input),
        "outputs": graph_shapes(model.graph.output),
        "nodes": len(model.graph.node),
        **artifact_metadata(output_path),
    }
    if args.ort_check:
        report["ort_disable_all"] = output_errors(
            reference,
            run_ort_outputs(output_path, inputs),
        )
    return report


def export_float_variant(
    name,
    spec,
    args,
    output_root,
    *,
    graph_mode,
    reparameterize,
):
    model, output_names, max_text_length = build_float_export_model(
        name,
        spec,
        reparameterize,
        graph_mode,
    )
    inputs, input_names = make_example_inputs(
        spec,
        args.batch_size,
        max_text_length=max_text_length,
        include_rec_targets=graph_mode == "training",
    )
    with torch.no_grad():
        reference = tensor_outputs(model(*inputs))

    reparameterization = (
        "reparameterized" if reparameterize else "non_reparameterized"
    )
    output_path = (
        output_root
        / name
        / f"{name}_{graph_mode}_{reparameterization}_float.onnx"
    )
    onnx_model = export_onnx(
        model,
        inputs,
        output_path,
        output_names,
        input_names=input_names,
        optimize=args.onnx_optimize,
    )
    report = {
        "format": "float",
        "graph": graph_mode,
        "reparameterized": reparameterize,
        "output": str(output_path.resolve()),
        "file_size": output_path.stat().st_size,
        "inputs": graph_shapes(onnx_model.graph.input),
        "outputs": graph_shapes(onnx_model.graph.output),
        "nodes": len(onnx_model.graph.node),
        "batch_normalization": sum(
            node.op_type == "BatchNormalization"
            for node in onnx_model.graph.node
        ),
    }
    if args.ort_check:
        report["ort_disable_all"] = output_errors(
            reference,
            run_ort_outputs(output_path, inputs),
        )
    return report


def export_quantonnx(name, spec, args, output_root):
    wrapper, output_names, max_text_length = build_quant_export_model(
        name,
        spec,
        args.quant_reparameterize,
        args.quant_graph,
    )
    inputs, input_names = make_example_inputs(
        spec,
        args.batch_size,
        max_text_length=max_text_length,
        include_rec_targets=args.quant_graph == "training",
    )
    prepared, float_nodes = prepare_qat_model(
        wrapper,
        inputs,
        load_axera_quantizer(spec.qat_config),
    )
    regional_targets = validate_regional_targets(prepared, spec.qat_config)
    weight_observers = initialize_weight_observers(prepared)
    prepared_qparams = prepared_activation_qparams(prepared)
    require_default_qparams(prepared_qparams, "Prepared graph")
    prepared_outputs = prepared_random_stage_outputs(prepared, inputs)
    converted = convert_prepared_model(prepared)
    with torch.no_grad():
        reference = tensor_outputs(converted(*inputs))

    output_path = (
        output_root
        / "quantonnx"
        / f"{name}_{args.quant_graph}_init_qat.onnx"
    )
    model = export_onnx(
        converted,
        inputs,
        output_path,
        output_names,
        input_names=input_names,
        optimize=args.onnx_optimize,
    )
    qdq = validate_qdq_graph(model)
    onnx_qparams = onnx_activation_qparams(model)
    require_default_qparams(onnx_qparams, "QuantONNX graph")
    report = {
        "format": "quantonnx",
        "graph": args.quant_graph,
        "auxiliary_branches": args.quant_graph == "training",
        "reparameterized": args.quant_reparameterize,
        "observer_state": "initialized",
        "training_steps": 0,
        "output": str(output_path.resolve()),
        "inputs": graph_shapes(model.graph.input),
        "outputs": graph_shapes(model.graph.output),
        "float_nodes": float_nodes,
        "prepared_nodes": len(list(prepared.graph.nodes)),
        "regional_targets": regional_targets,
        "converted_nodes": len(list(converted.graph.nodes)),
        "weight_observers_initialized": len(weight_observers),
        "activation_observers": len(prepared_qparams),
        "activation_qparams_default": True,
        "qdq": qdq,
        **artifact_metadata(output_path),
    }
    ort_outputs = None
    if args.ort_check:
        ort_outputs = run_ort_outputs(output_path, inputs)
        report["ort_disable_all"] = output_errors(
            reference,
            ort_outputs,
        )
    report["random_stage_comparison"] = random_stage_comparison(
        prepared_outputs,
        reference,
        output_names,
        ort_outputs,
    )
    return report


def export_quant_variant(
    name,
    spec,
    args,
    output_root,
    *,
    graph_mode,
    reparameterize,
):
    wrapper, output_names, max_text_length = build_quant_export_model(
        name,
        spec,
        reparameterize,
        graph_mode,
    )
    inputs, input_names = make_example_inputs(
        spec,
        args.batch_size,
        max_text_length=max_text_length,
        include_rec_targets=graph_mode == "training",
    )
    prepared, float_nodes = prepare_qat_model(
        wrapper,
        inputs,
        load_axera_quantizer(spec.qat_config),
    )
    regional_targets = validate_regional_targets(prepared, spec.qat_config)
    weight_observers = initialize_weight_observers(prepared)
    prepared_qparams = prepared_activation_qparams(prepared)
    require_default_qparams(prepared_qparams, "Prepared graph")
    prepared_outputs = prepared_random_stage_outputs(prepared, inputs)
    converted = convert_prepared_model(prepared)
    with torch.no_grad():
        reference = tensor_outputs(converted(*inputs))

    reparameterization = (
        "reparameterized" if reparameterize else "non_reparameterized"
    )
    output_path = (
        output_root
        / name
        / f"{name}_{graph_mode}_{reparameterization}_u16s16_qat.onnx"
    )
    model = export_onnx(
        converted,
        inputs,
        output_path,
        output_names,
        input_names=input_names,
        optimize=args.onnx_optimize,
    )
    qdq = qdq_stats(model)
    standard_qdq_contract = qdq["batch_normalization"] == 0
    if standard_qdq_contract:
        validate_qdq_graph(model)
    ort_outputs = None
    if args.ort_check:
        ort_outputs = run_ort_outputs(output_path, inputs)
    report = {
        "format": "quantonnx",
        "graph": graph_mode,
        "reparameterized": reparameterize,
        "quantization": "global U16 activation + S16 weight; Attention S16",
        "output": str(output_path.resolve()),
        "inputs": graph_shapes(model.graph.input),
        "outputs": graph_shapes(model.graph.output),
        "float_nodes": float_nodes,
        "prepared_nodes": len(list(prepared.graph.nodes)),
        "regional_targets": regional_targets,
        "converted_nodes": len(list(converted.graph.nodes)),
        "weight_observers_initialized": len(weight_observers),
        "activation_observers": len(prepared_qparams),
        "activation_qparams_default": True,
        "qdq": qdq,
        "axera_qdq_contract": "pass" if standard_qdq_contract else "blocked_by_batch_normalization",
        "file_size": output_path.stat().st_size,
    }
    if args.ort_check:
        report["ort_disable_all"] = output_errors(
            reference,
            ort_outputs,
        )
    report["random_stage_comparison"] = random_stage_comparison(
        prepared_outputs,
        reference,
        output_names,
        ort_outputs,
    )
    return report


def audit_existing_artifacts(report, model_names):
    audited = {}
    for name in model_names:
        model_record = report.get("models", {}).get(name)
        if model_record is None:
            raise ValueError(f"Report does not contain model: {name}")
        audited[name] = {}
        for format_name in ("float", "quantonnx", "quantonnx_inference"):
            format_record = model_record.get(format_name)
            if format_record is None:
                continue
            output_path = Path(format_record["output"])
            model = onnx.load(str(output_path), load_external_data=True)
            onnx.checker.check_model(model, full_check=True)
            format_record.update(artifact_metadata(output_path))
            audited[name][format_name] = {
                "output": str(output_path.resolve()),
                "file_size": format_record["file_size"],
                "nodes": len(model.graph.node),
            }
    return audited


def load_training_report(output_root):
    report_path = output_root / "training_onnx_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("contract") != "complete_pretrained_training_graph":
            raise ValueError(f"Unexpected existing report contract: {report_path}")
    else:
        report = {
            "contract": "complete_pretrained_training_graph",
            "models": {},
        }
    return report, report_path


def run_training_export(args):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    output_root = Path(args.output_dir)
    report, report_path = load_training_report(output_root)
    report.update(
        {
            "float_reparameterized": args.float_reparameterize,
            "quant_reparameterized": args.quant_reparameterize,
            "quant_graph": args.quant_graph,
            "onnx_optimized": args.onnx_optimize,
        }
    )
    for name in args.models:
        spec = MODEL_SPECS[name]
        record = report["models"].setdefault(name, {})
        record.update(
            {
                "task": spec.task,
                "model_config": str(Path(spec.model_config).resolve()),
                "weights": str(Path(spec.weights).resolve()),
                "qat_config": str(Path(spec.qat_config).resolve()),
            }
        )
        if args.format in ("float", "both"):
            record["float"] = export_float(name, spec, args, output_root)
        if args.format in ("quantonnx", "both"):
            record_key = (
                "quantonnx"
                if args.quant_graph == "training"
                else "quantonnx_inference"
            )
            record[record_key] = export_quantonnx(name, spec, args, output_root)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({name: record}, indent=2, sort_keys=True))


def run_float_matrix_export(args):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    output_root = Path(args.output_dir)
    report = {
        "contract": "float_training_inference_reparameterization_matrix",
        "batch_size": args.batch_size,
        "onnx_optimized": args.onnx_optimize,
        "ort_disable_all_checked": args.ort_check,
        "models": {},
    }
    for name in args.models:
        spec = MODEL_SPECS[name]
        model_record = {
            "task": spec.task,
            "model_config": str(Path(spec.model_config).resolve()),
            "weights": str(Path(spec.weights).resolve()),
            "variants": {},
        }
        report["models"][name] = model_record
        for graph_mode in args.graphs:
            for reparameterization in args.reparameterizations:
                reparameterize = reparameterization == "reparameterized"
                variant_name = f"{graph_mode}_{reparameterization}"
                model_record["variants"][variant_name] = export_float_variant(
                    name,
                    spec,
                    args,
                    output_root,
                    graph_mode=graph_mode,
                    reparameterize=reparameterize,
                )
                print(
                    json.dumps(
                        {
                            "model": name,
                            "variant": variant_name,
                            "result": model_record["variants"][variant_name],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
        report_path = output_root / "float_onnx_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


QUANT_MATRIX_CONFIGS = {
    "ppocrv5_mobile_det": {
        True: "configs/qat/ppocrv5_mobile_det_u16s16_reparameterized.json",
        False: "configs/qat/ppocrv5_mobile_det_u16s16_non_reparameterized.json",
    },
    "ppocrv5_mobile_rec": {
        True: "configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16_reparameterized.json",
        False: "configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16_non_reparameterized.json",
    },
    "ppocrv6_small_det": {
        True: "configs/qat/ppocrv6_small_det_u16s16_reparameterized.json",
        False: "configs/qat/ppocrv6_small_det_u16s16_non_reparameterized.json",
    },
    "ppocrv6_small_rec": {
        True: "configs/qat/ppocrv6_small_rec_u16s16_attn_s16_reparameterized.json",
        False: "configs/qat/ppocrv6_small_rec_u16s16_attn_s16_non_reparameterized.json",
    },
}


def run_quant_matrix_export(args):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    output_root = Path(args.output_dir)
    report = {
        "contract": "u16s16_quantonnx_training_inference_reparameterization_matrix",
        "batch_size": args.batch_size,
        "onnx_optimized": args.onnx_optimize,
        "ort_disable_all_checked": args.ort_check,
        "models": {},
    }
    for name in args.models:
        spec = MODEL_SPECS[name]
        model_record = {
            "task": spec.task,
            "model_config": str(Path(spec.model_config).resolve()),
            "weights": str(Path(spec.weights).resolve()),
            "variants": {},
        }
        report["models"][name] = model_record
        for graph_mode in args.graphs:
            for reparameterization in args.reparameterizations:
                reparameterize = reparameterization == "reparameterized"
                variant_name = f"{graph_mode}_{reparameterization}"
                variant_spec = dataclasses.replace(
                    spec,
                    qat_config=QUANT_MATRIX_CONFIGS[name][reparameterize],
                )
                model_record["variants"][variant_name] = export_quant_variant(
                    name,
                    variant_spec,
                    args,
                    output_root,
                    graph_mode=graph_mode,
                    reparameterize=reparameterize,
                )
                print(
                    json.dumps(
                        {
                            "model": name,
                            "variant": variant_name,
                            "result": model_record["variants"][variant_name],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
        report_path = output_root / "quantonnx_matrix_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def run_training_audit(args):
    output_root = Path(args.output_dir)
    report, report_path = load_training_report(output_root)
    audited = audit_existing_artifacts(report, args.models)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audited, indent=2, sort_keys=True))


def run_initialized_export(args):
    torch.manual_seed(20260805)
    if args.dynamic_heights and args.task != "rec":
        raise ValueError("--dynamic-heights is only valid for recognition.")
    model = build_task_model(
        args.task,
        args.model_config,
        weights_path=args.weights,
        reparameterize=args.reparameterize,
        det_graph=args.det_graph,
    )
    example = torch.empty([1, *args.image_shape], dtype=torch.float32)
    dynamic_shapes = build_qat_dynamic_shapes(
        example,
        dynamic_heights=args.dynamic_heights,
    )
    prepared, float_node_count = prepare_qat_model(
        model,
        (example,),
        load_axera_quantizer(args.qat_config),
        dynamic_shapes=dynamic_shapes,
    )
    regional_targets = validate_regional_targets(prepared, args.qat_config)
    weight_observers = initialize_weight_observers(prepared)
    prepared_qparams = prepared_activation_qparams(prepared)
    require_default_qparams(prepared_qparams, "Prepared graph")
    prepared_outputs = prepared_random_stage_outputs(prepared, (example,))

    converted = convert_prepared_model(prepared)
    output_index = 0 if args.task == "det" and args.det_graph == "training" else None
    with torch.no_grad():
        converted_outputs = selected_outputs(converted(example), output_index)
    onnx_model = export_onnx(
        converted,
        (example,),
        args.output,
        ["maps" if args.task == "det" else "logits"],
        optimize=args.onnx_optimize,
        output_index=output_index,
        static_axis_values={
            0: {
                1: args.image_shape[0],
                2: args.image_shape[1],
                3: args.image_shape[2],
            }
        },
    )
    qdq = validate_qdq_graph(onnx_model)
    onnx_qparams = onnx_activation_qparams(onnx_model)
    require_default_qparams(onnx_qparams, "QuantONNX graph")
    prepared_outputs = {
        stage: selected_outputs(outputs, output_index)
        for stage, outputs in prepared_outputs.items()
    }
    ort_outputs = (run_ort(args.output, example, optimize=False),)

    report = {
        "task": args.task,
        "model_config": str(Path(args.model_config).resolve()),
        "weights": str(Path(args.weights).resolve()),
        "qat_config": str(Path(args.qat_config).resolve()),
        "output": str(Path(args.output).resolve()),
        "image_shape": [1, *args.image_shape],
        "prepared_dynamic_heights": args.dynamic_heights or [],
        "quantonnx_static_image_shape": args.image_shape,
        "reparameterized": args.reparameterize,
        "onnx_optimized": args.onnx_optimize,
        "inference_runs": 0,
        "calibration_runs": 0,
        "training_steps": 0,
        "float_nodes": float_node_count,
        "prepared_nodes": len(list(prepared.graph.nodes)),
        "regional_targets": regional_targets,
        "converted_nodes": len(list(converted.graph.nodes)),
        "weight_observers_initialized": len(weight_observers),
        "prepared_activation_observers": len(prepared_qparams),
        "prepared_activation_qparams_default": True,
        "onnx_quantize_linear": len(onnx_qparams),
        "onnx_activation_qparams_default": True,
        "qdq": qdq,
        "inputs": graph_shapes(onnx_model.graph.input),
        "random_stage_comparison": random_stage_comparison(
            prepared_outputs,
            converted_outputs,
            ("maps" if args.task == "det" else "logits",),
            ort_outputs,
        ),
        **artifact_metadata(args.output),
    }
    report_path = Path(args.report or f"{args.output}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def value_from_args_or_metadata(args, metadata, name):
    argument_value = getattr(args, name)
    if argument_value is not None:
        return argument_value
    value = metadata.get(name)
    if value is None:
        option = name.replace("_", "-")
        raise ValueError(f"Checkpoint metadata is missing --{option}.")
    return value


def run_checkpoint_export(args):
    torch.manual_seed(20260728)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    if not metadata.get("qat", False):
        raise ValueError("Checkpoint is not marked as an Axera QAT checkpoint.")
    task = value_from_args_or_metadata(args, metadata, "task")
    model_config = relocate_project_path(
        value_from_args_or_metadata(args, metadata, "model_config")
    )
    qat_config = relocate_project_path(
        value_from_args_or_metadata(args, metadata, "qat_config")
    )
    image_shape = tuple(
        args.image_shape
        if args.image_shape is not None
        else metadata.get("image_shape", [])
    )
    if len(image_shape) != 3:
        raise ValueError("Checkpoint metadata is missing a valid image shape.")
    captured_batch = int(metadata.get("batch_size", 1))
    dynamic_batch = (
        bool(metadata.get("dynamic_batch", False))
        if args.dynamic_batch is None
        else args.dynamic_batch
    )
    if not dynamic_batch and args.batch_size != captured_batch:
        raise ValueError(
            "This checkpoint has a static batch dimension; export with "
            f"--batch-size {captured_batch}."
        )

    build_metadata = {
        **metadata,
        "task": task,
        "model_config": model_config,
        "qat_config": qat_config,
        "image_shape": list(image_shape),
        "batch_size": captured_batch,
        "dynamic_batch": dynamic_batch,
    }
    prepared, float_node_count = build_prepared_qat_checkpoint(
        build_metadata,
        weights_path=args.weights,
    )
    prepared.load_state_dict(checkpoint["model"], strict=True)
    prepared_node_count = len(list(prepared.graph.nodes))
    export_images = torch.randn(
        [args.batch_size, *image_shape],
        dtype=torch.float32,
    )
    full_rec_graph = (
        task == "rec" and metadata.get("rec_graph") == "pretrained_train"
    )
    comparison_inputs = (export_images,)
    max_text_length = None
    if full_rec_graph:
        max_text_length = int(
            load_ocr_config(model_config)["Global"].get("max_text_length", 25)
        )
        comparison_inputs = (
            export_images,
            torch.zeros(
                args.batch_size,
                max_text_length,
                dtype=torch.int64,
            ),
        )
    prepared_outputs = prepared_random_stage_outputs(prepared, comparison_inputs)
    converted = convert_prepared_model(prepared)
    del prepared
    gc.collect()
    dynamic_heights = list(metadata.get("dynamic_heights") or [])

    onnx_dynamic_shapes = build_qat_dynamic_shapes(
        export_images,
        dynamic_batch=dynamic_batch,
    )
    with torch.no_grad():
        converted_outputs = converted(*comparison_inputs)
    output_index = 0 if task == "det" or full_rec_graph else None
    output_names = ("maps" if task == "det" else "logits",)
    reference = (
        converted_outputs[output_index]
        if output_index is not None
        else converted_outputs
    )
    converted_comparison_outputs = selected_outputs(
        converted_outputs,
        output_index,
    )
    prepared_outputs = {
        stage: selected_outputs(outputs, output_index)
        for stage, outputs in prepared_outputs.items()
    }
    export_model = converted
    export_output_index = output_index
    if full_rec_graph:
        export_model = RecTrainingDeploymentProjection(
            converted,
            max_text_length=max_text_length,
        )
        export_output_index = None
    onnx_model = export_onnx(
        export_model,
        (export_images,),
        args.output,
        output_names,
        optimize=args.onnx_optimize,
        output_index=export_output_index,
        dynamic_shapes=onnx_dynamic_shapes,
        static_axis_values={
            0: {1: image_shape[0], 2: image_shape[1], 3: image_shape[2]}
        },
    )
    onnx_stats = validate_qdq_graph(onnx_model)
    if full_rec_graph and len(onnx_model.graph.input) != 1:
        raise RuntimeError(
            "Recognition deployment QuantONNX must contain only the images input."
        )
    ort_output = run_ort(args.output, export_images, optimize=False)
    optimized_ort_output = (
        run_ort(args.output, export_images, optimize=True)
        if args.ort_optimizer_check
        else None
    )
    error = numpy_error_stats(reference.detach().cpu().numpy(), ort_output)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output": str(Path(args.output).resolve()),
        "task": task,
        "dynamic_batch": dynamic_batch,
        "prepared_dynamic_heights": dynamic_heights,
        "quantonnx_static_image_shape": list(image_shape),
        "onnx_optimized": args.onnx_optimize,
        "float_nodes": float_node_count,
        "prepared_nodes": prepared_node_count,
        "converted_nodes": len(list(converted.graph.nodes)),
        "inputs": graph_shapes(onnx_model.graph.input),
        "outputs": graph_shapes(onnx_model.graph.output),
        "onnx": onnx_stats,
        "pytorch_ort_mae": error["mae"],
        "pytorch_ort_max_abs": error["max_abs"],
        "output_finite": bool(np.isfinite(ort_output).all()),
        "random_stage_comparison": random_stage_comparison(
            prepared_outputs,
            converted_comparison_outputs,
            output_names,
            (ort_output,),
        ),
        **artifact_metadata(args.output),
    }
    if optimized_ort_output is not None:
        optimizer_error = numpy_error_stats(ort_output, optimized_ort_output)
        result["ort_optimizer_mae"] = optimizer_error["mae"]
        result["ort_optimizer_max_abs"] = optimizer_error["max_abs"]
    reference_output = None
    if args.onnx_reference:
        reference_output = run_onnx_reference(onnx_model, export_images)
        reference_error = numpy_error_stats(
            reference.detach().cpu().numpy(),
            reference_output,
        )
        result["pytorch_onnx_reference_mae"] = reference_error["mae"]
        result["pytorch_onnx_reference_max_abs"] = reference_error["max_abs"]
    if task == "rec":
        torch_array = reference.detach().cpu().numpy()
        if reference_output is not None:
            result["rec_onnx_reference"] = sequence_error_stats(
                torch_array,
                reference_output,
            )
        result["rec_ort"] = sequence_error_stats(torch_array, ort_output)
        if optimized_ort_output is not None:
            result["rec_ort_optimized"] = sequence_error_stats(
                torch_array,
                optimized_ort_output,
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.handler(arguments)
