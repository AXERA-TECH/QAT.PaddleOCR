#!/usr/bin/env python3
"""Discover PP-OCR det/rec FX regions and generate an Axera QAT config."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from pytorchocr.training import build_task_model, load_ocr_config


LINEAR_OPS = {torch.ops.aten.linear.default}
MATMUL_OPS = {torch.ops.aten.matmul.default}
MUL_OPS = {torch.ops.aten.mul.Tensor}
SOFTMAX_OPS = {torch.ops.aten.softmax.int}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("auto", "det", "rec"), default="auto")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--image-shape", nargs=3, type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--expected-attention",
        type=int,
        help="Optional exact Attention-region count; zero is valid for det models.",
    )
    parser.add_argument(
        "--attention-dtype",
        choices=("auto", "S8", "S16"),
        default="auto",
        help=(
            "Signed activation dtype inside Attention. auto maps global U8 to S8 "
            "and global U16 to S16; use an explicit value for mixed-width regions."
        ),
    )
    parser.add_argument(
        "--reparameterize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--keep-bn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the native post-sum BN of v6 RepDWConv during "
            "reparameterization (matches train.py --keep-bn QAT contract)."
        ),
    )
    parser.add_argument(
        "--rec-graph",
        choices=("deploy", "pretrained_train"),
        default="deploy",
        help=(
            "Recognition graph for FX discovery. deploy matches the inference "
            "QuantONNX; pretrained_train matches the full training graph "
            "(FullRecTrainingWrapper with gtc targets) used by QAT training."
        ),
    )
    parser.add_argument(
        "--dynamic-batch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Match the training dynamic-batch contract. Defaults to True when "
            "--rec-graph pretrained_train (FullRecTrainingWrapper needs "
            "batch-aligned inputs), else False. The QAT JSON module names are "
            "discovered from a graph prepared with the same dynamic-shape "
            "contract as training, because node numbering shifts when dynamic "
            "shapes are enabled."
        ),
    )
    parser.add_argument(
        "--no-proj-entry",
        action="store_true",
        help=(
            "Do not emit the output-projection Linear regional entry. The "
            "projection (mixer.proj) is FC-like; leaving it on the global "
            "domain avoids requantize boundaries with the second MatMul "
            "output (softmax·V -> proj). Default: emit the proj entry."
        ),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--strict-names",
        action="store_true",
        help=(
            "With --check, additionally require every module name of the "
            "template (not only the regenerated entries) to exist in the "
            "current prepared graph. Catches checked-in configs whose "
            "hand-written entries (e.g. v6 gtc branch) were named against a "
            "different graph contract than the one being prepared. Default "
            "off: templates may legitimately be unions over several graph "
            "forms (training + folded + smoke)."
        ),
    )
    return parser.parse_args()


def module_entries(node: torch.fx.Node) -> list[tuple[str, str]]:
    stack = node.meta.get("nn_module_stack") or {}
    return [
        (str(value[0]), str(value[1]))
        for value in stack.values()
        if isinstance(value, tuple) and len(value) >= 2
    ]


def attention_owner(node: torch.fx.Node) -> str | None:
    for path, module_type in reversed(module_entries(node)):
        if module_type.endswith(".Attention"):
            return path
    return None


def module_paths(node: torch.fx.Node) -> list[str]:
    return [path for path, _ in module_entries(node)]


def node_inputs(value: Any):
    if isinstance(value, torch.fx.Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from node_inputs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from node_inputs(item)


def direct_node_inputs(node: torch.fx.Node) -> list[torch.fx.Node]:
    return list(node_inputs((node.args, node.kwargs)))


def exactly_one(nodes: list[torch.fx.Node], role: str, owner: str) -> torch.fx.Node:
    if len(nodes) != 1:
        raise RuntimeError(f"{owner} expected one {role}, found {[node.name for node in nodes]}")
    return nodes[0]


def discover(
    gm: torch.fx.GraphModule,
    expected_attention: int | None,
) -> dict[str, Any]:
    nodes = list(gm.graph.nodes)
    order = {node: position for position, node in enumerate(nodes)}
    groups: dict[str, list[torch.fx.Node]] = defaultdict(list)
    for node in nodes:
        owner = attention_owner(node)
        if owner is not None:
            groups[owner].append(node)

    regions = []
    for owner, owned in groups.items():
        qkv = exactly_one(
            [
                node
                for node in owned
                if node.op == "call_function"
                and node.target in LINEAR_OPS
                and any(path.endswith(".mixer.qkv") for path in module_paths(node))
            ],
            "QKV Linear",
            owner,
        )
        proj = exactly_one(
            [
                node
                for node in owned
                if node.op == "call_function"
                and node.target in LINEAR_OPS
                and any(path.endswith(".mixer.proj") for path in module_paths(node))
            ],
            "output projection Linear",
            owner,
        )
        matmuls = sorted(
            [node for node in owned if node.op == "call_function" and node.target in MATMUL_OPS],
            key=order.__getitem__,
        )
        if len(matmuls) != 2:
            raise RuntimeError(f"{owner} expected two MatMul nodes, found {[node.name for node in matmuls]}")
        first, second = matmuls
        softmax = exactly_one(
            [node for node in owned if node.op == "call_function" and node.target in SOFTMAX_OPS],
            "Softmax",
            owner,
        )
        scale_mul = exactly_one(
            [
                node
                for node in owned
                if node.op == "call_function"
                and node.target in MUL_OPS
                and order[qkv] < order[node] < order[first]
            ],
            "Q scale Mul",
            owner,
        )
        first_inputs = set(direct_node_inputs(first))
        second_inputs = set(direct_node_inputs(second))
        if scale_mul not in first_inputs:
            raise RuntimeError(f"{owner} scale Mul does not feed the first MatMul")
        if first not in set(direct_node_inputs(softmax)):
            raise RuntimeError(f"{owner} first MatMul does not feed Softmax directly")
        if softmax not in second_inputs:
            # Training-mode dropout may remain between Softmax and second MatMul.
            if not any(
                input_node.target == torch.ops.aten.dropout.default
                and softmax in direct_node_inputs(input_node)
                for input_node in second_inputs
                if input_node.op == "call_function"
            ):
                raise RuntimeError(f"{owner} Softmax does not feed the second MatMul")
        regions.append(
            {
                "owner": owner,
                "qkv_linear": qkv.name,
                "proj_linear": proj.name,
                "scale_mul": scale_mul.name,
                "first_matmul": first.name,
                "softmax": softmax.name,
                "second_matmul": second.name,
                "source_fn_stack": {
                    role: [str(item) for item in node.meta.get("source_fn_stack", [])]
                    for role, node in {
                        "qkv_linear": qkv,
                        "proj_linear": proj,
                        "scale_mul": scale_mul,
                        "first_matmul": first,
                        "softmax": softmax,
                        "second_matmul": second,
                    }.items()
                },
            }
        )
    regions.sort(key=lambda item: item["owner"])
    if expected_attention is not None and len(regions) != expected_attention:
        raise RuntimeError(
            f"Expected {expected_attention} complete Attention regions, "
            f"found {len(regions)}: {[item['owner'] for item in regions]}"
        )
    return {"attention": regions, "fx_nodes": len(nodes)}


def signed_qspec(dtype: str) -> dict[str, Any]:
    if dtype == "S8":
        return {"dtype": "S8", "qmin": -127, "qmax": 127}
    if dtype == "S16":
        return {"dtype": "S16", "qmin": -32767, "qmax": 32767}
    raise ValueError(f"Unsupported signed activation dtype: {dtype}")


def resolve_attention_dtype(
    requested: str,
    global_activation: dict[str, Any],
) -> str:
    if requested != "auto":
        signed_qspec(requested)
        return requested
    activation_dtype = global_activation.get("dtype")
    inferred = {"U8": "S8", "U16": "S16"}.get(activation_dtype)
    if inferred is None:
        raise RuntimeError(
            f"Cannot infer Attention dtype from global activation {activation_dtype!r}"
        )
    return inferred


def validate_global_qspec(
    global_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    activation = copy.deepcopy(global_config.get("input", {}))
    weight = copy.deepcopy(global_config.get("weight", {}))
    expected_ranges = {
        "U8": (0, 255),
        "U16": (0, 65535),
        "S8": (-127, 127),
        "S16": (-32767, 32767),
    }
    for role, qspec, allowed in (
        ("activation", activation, {"U8", "U16"}),
        ("weight", weight, {"S8", "S16"}),
    ):
        dtype = qspec.get("dtype")
        if dtype not in allowed:
            raise RuntimeError(
                f"PP-OCR global {role} dtype must be one of {sorted(allowed)}"
            )
        if (qspec.get("qmin"), qspec.get("qmax")) != expected_ranges[dtype]:
            raise RuntimeError(f"Invalid {dtype} qmin/qmax for global {role}")
    return activation, weight


def update_config(
    template: dict[str, Any],
    discovery: dict[str, Any],
    attention_dtype: str,
    include_proj: bool = True,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    global_config = config.get("global_config", {})
    global_activation, global_weight = validate_global_qspec(global_config)

    regions = discovery["attention"]
    if not regions:
        config["regional_configs"] = []
        return config
    resolved_attention_dtype = resolve_attention_dtype(
        attention_dtype,
        global_activation,
    )
    signed = signed_qspec(resolved_attention_dtype)
    regional = [
        {
             "module_names": [item["qkv_linear"] for item in regions],
             "module_type": "linear",
             "module_config": {
                 "is_symmetric": False,
                 "output_is_symmetric": True,
                 "input": global_activation,
                 "output": signed,
                 "weight": global_weight,
             },
         },
    ]
    if include_proj:
        regional.append(
            {
                "module_names": [item["proj_linear"] for item in regions],
                "module_type": "linear",
                "module_config": {
                    "is_symmetric": True,
                    "input": signed,
                    "weight": global_weight,
                },
            }
        )
    regional.extend(
        [
        {
             "module_names": [item["scale_mul"] for item in regions],
             "module_type": "mul",
             "module_config": {
                 "is_symmetric": True,
                 "output_is_symmetric": True,
                 "input": signed,
                 "output": signed,
             },
         },
         {
             "module_names": [item["first_matmul"] for item in regions],
             "module_type": "matmul",
             "module_config": {
                 "is_symmetric": True,
                 "output_is_symmetric": True,
                 "input": signed,
                 "output": signed,
             },
         },
         {
             "module_names": [item["softmax"] for item in regions],
             "module_type": "softmax",
             "module_config": {
                 "is_symmetric": True,
                 "output_is_symmetric": True,
                 "input": signed,
                 "output": signed,
             },
         },
         {
             "module_names": [item["second_matmul"] for item in regions],
             "module_type": "matmul",
             "module_config": {
                 "is_symmetric": True,
                 "input": signed,
             },
         },
        ]
    )
    config["regional_configs"] = regional
    return config


def _config_matches_template(generated: dict[str, Any], template: dict[str, Any]) -> bool:
    """Union-aware --check comparison.

    Checked-in QAT JSON may carry module names from several graph forms
    (training graph + folded graph + smoke graph, see training record §39).
    Regenerating from one graph form can therefore never equal such a template.
    Accept the generated config when each generated entry is subsumed by a
    template entry of the same module_type: equal module_config and every
    generated name present in the template's name list. Template entries that
    have no generated counterpart (other graph forms, e.g. the S16 downsampling
    entries) are allowed and must be verified by annotation-level inspection of
    the target graph instead.
    """
    if generated.get("global_config") != template.get("global_config"):
        return False
    global_activation, _ = validate_global_qspec(template.get("global_config", {}))
    by_type: dict[str, list[dict[str, Any]]] = {}
    for entry in template.get("regional_configs", []):
        by_type.setdefault(entry.get("module_type"), []).append(entry)
    for entry in generated.get("regional_configs", []):
        module_type = entry.get("module_type")
        generated_names = set(entry.get("module_names") or [])
        candidates = by_type.get(module_type, [])
        if not any(
            _module_config_equivalent(
                entry.get("module_config"),
                candidate.get("module_config"),
                global_activation,
            )
            and generated_names <= set(candidate.get("module_names") or [])
            for candidate in candidates
        ):
            return False
    return True


def _module_config_equivalent(
    generated: dict[str, Any],
    template: dict[str, Any],
    global_activation: dict[str, Any],
) -> bool:
    """Compare regional module_configs, treating an omitted ``output`` as
    equivalent to an explicit global-activation output.

    The softmax . V MatMul entry intentionally omits ``output`` so the result
    domain follows the global activation (U8). Older configs (e.g. the v5-rec
    attention S8 file) spell the same choice out explicitly as
    ``output == global_activation`` with ``output_is_symmetric: False``.
    Normalize both sides by stripping such redundant explicit output fields
    before comparing so --check accepts both forms.
    """
    if generated == template:
        return True

    def normalized(cfg: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(cfg, dict):
            return cfg
        norm = dict(cfg)
        output = norm.pop("output", None)
        symmetric = norm.pop("output_is_symmetric", None)
        if output == global_activation and symmetric is False:
            return norm
        if output is not None:
            norm["output"] = output
        if symmetric is not None:
            norm["output_is_symmetric"] = symmetric
        return norm

    return normalized(generated) == normalized(template)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.expected_attention is not None and args.expected_attention < 0:
        raise ValueError("--expected-attention must be non-negative")
    if not args.check and args.output is None:
        raise ValueError("--output is required unless --check is used")
    if args.output is not None:
        if args.output.resolve() == args.base_config.resolve():
            raise ValueError("--output must not overwrite --base-config")
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    model_config = load_ocr_config(args.model_config)
    detected_task = model_config["Architecture"].get("model_type")
    if detected_task not in ("det", "rec"):
        raise ValueError(f"Unsupported PP-OCR model_type: {detected_task}")
    if args.task != "auto" and args.task != detected_task:
        raise ValueError(
            f"--task {args.task} does not match model config task {detected_task}"
        )
    task = detected_task
    model = build_task_model(
        task,
        args.model_config,
        weights_path=args.weights,
        reparameterize=args.reparameterize,
        det_graph="inference",
        rec_graph=args.rec_graph,
        keep_bn=args.keep_bn,
    )
    images = torch.zeros([args.batch_size, *args.image_shape], dtype=torch.float32)
    example_inputs: tuple[Any, ...] = (images,)
    if task == "rec" and args.rec_graph == "pretrained_train":
        from pytorchocr.quantization import FullRecTrainingWrapper

        max_text_length = int(model_config["Global"].get("max_text_length", 25))
        model = FullRecTrainingWrapper(
            model, max_text_length=max_text_length
        ).set_qat_capture_mode()
        example_inputs = (
            images,
            torch.zeros(
                [args.batch_size, max_text_length],
                dtype=torch.int64,
            ),
        )
    dynamic_batch = (
        args.batch_size > 1
        if args.dynamic_batch is None
        else args.dynamic_batch
    )
    batch_aligned = int(task == "rec" and args.rec_graph == "pretrained_train")
    export_dynamic_shapes = None
    if dynamic_batch:
        from pytorchocr.quantization import build_qat_dynamic_shapes

        export_dynamic_shapes = build_qat_dynamic_shapes(
            images,
            dynamic_batch=True,
            dynamic_heights=None,
            batch_aligned_inputs=batch_aligned,
            max_batch=args.batch_size,
        )
    gm = torch.export.export_for_training(
        model, example_inputs, dynamic_shapes=export_dynamic_shapes
    ).module()

    # Discover from the *prepared* graph. QAT regional module names are matched
    # against the graph produced by prepare_qat_pt2e; node numbering shifts
    # with the training contract (dynamic batch / batch-aligned gtc targets /
    # batch size / model wrapper), so a discovery on the raw export graph may
    # miss entries (e.g. mul_58 vs mul_352). Prepare the ORIGINAL model (not
    # the already-exported gm: nested export changes numbering) with the same
    # contract as train.py, then remap discovered names onto it.
    from pytorchocr.quantization import (
        build_qat_dynamic_shapes,
        load_axera_quantizer,
        prepare_qat_model,
    )

    prepared, _ = prepare_qat_model(
        model,
        example_inputs,
        load_axera_quantizer(args.base_config),
        dynamic_shapes=build_qat_dynamic_shapes(
            images,
            dynamic_batch=dynamic_batch,
            dynamic_heights=None,
            batch_aligned_inputs=batch_aligned,
            max_batch=args.batch_size,
        ),
    )
    discovery = discover(gm, args.expected_attention)
    # Node numbering shifts between the raw export graph and the prepared
    # training graph (dynamic shapes insert nodes). Remap every discovered
    # node to its counterpart in the prepared graph by matching the same
    # call op under the same module stack.
    prepared_order: dict[str, int] = {}
    prepared_by_stack: dict[tuple[str, ...], list[torch.fx.Node]] = defaultdict(list)
    for position, node in enumerate(prepared.graph.nodes):
        if node.op != "call_function":
            continue
        prepared_order[node.name] = position
        stack = tuple(module_paths(node))
        prepared_by_stack[stack].append(node)

    def remap(role_node: torch.fx.Node, role: str) -> torch.fx.Node:
        target = role_node.target
        stack = tuple(module_paths(role_node))
        candidates = [
            node
            for node in prepared_by_stack.get(stack, [])
            if node.target == target
        ]
        if len(candidates) == 0:
            raise RuntimeError(
                f"Prepared graph counterpart missing for {role_node.name} "
                f"({target}, stack {stack})"
            )
        if len(candidates) == 1:
            return candidates[0]
        # Same op under the same module stack (e.g. two MatMul in one mixer):
        # disambiguate by graph position relative to the other roles of this
        # region. first_matmul precedes softmax; second_matmul follows it.
        if role == "first_matmul":
            softmax_node = gm_names[region["softmax"]]
            preceding = [
                node for node in candidates
                if prepared_order[node.name] < prepared_order[softmax_node.name]
            ]
            if len(preceding) == 1:
                return preceding[0]
        if role == "second_matmul":
            softmax_node = gm_names[region["softmax"]]
            following = [
                node for node in candidates
                if prepared_order[node.name] > prepared_order[softmax_node.name]
            ]
            if len(following) == 1:
                return following[0]
        raise RuntimeError(
            f"Prepared graph counterpart not unique for {role_node.name} "
            f"({target}, stack {stack}): {[c.name for c in candidates]}"
        )

    gm_names = {node.name: node for node in gm.graph.nodes}
    for region in discovery["attention"]:
        for role in ("qkv_linear", "proj_linear", "scale_mul", "first_matmul",
                     "softmax", "second_matmul"):
            region[role] = remap(gm_names[region[role]], role).name
    template = json.loads(args.base_config.read_text(encoding="utf-8"))
    global_activation, _ = validate_global_qspec(template.get("global_config", {}))
    resolved_attention_dtype = resolve_attention_dtype(
        args.attention_dtype,
        global_activation,
    )
    generated = update_config(
        template, discovery, resolved_attention_dtype,
        include_proj=not args.no_proj_entry,
    )

    # Defensive check: every generated regional name must exist in the same
    # prepared graph it was discovered from.
    prepared_names = {node.name for node in prepared.graph.nodes}
    missed = [
        name
        for entry in generated["regional_configs"]
        for name in (entry.get("module_names") or [])
        if name not in prepared_names
    ]
    if missed:
        raise SystemExit(
            "ERROR: QAT regional module names miss the prepared training "
            f"graph: {missed}. Regenerate with matching --batch-size/"
            "--dynamic-batch (node numbering shifts under dynamic shapes)."
        )
    print("QAT config names verified against prepared training graph")
    if args.check:
        if args.strict_names:
            missed_entries = [
                (entry.get("module_type"), entry.get("module_names"))
                for entry in template.get("regional_configs", [])
                if not any(
                    name in prepared_names
                    for name in (entry.get("module_names") or [])
                )
            ]
            if missed_entries:
                raise SystemExit(
                    "ERROR(--strict-names): template regional entries with no "
                    f"name in the current prepared graph: {missed_entries}. "
                    "The checked-in config was generated for another graph "
                    "contract (e.g. static batch 1 vs dynamic batch 64 shifts "
                    "Mul numbering under dynamic shapes). Regenerate for the "
                    "training contract and re-check. Note: entries are checked "
                    "per-entry (at least one name must hit); union entries "
                    "that mix names of several graph forms are allowed."
                )
        if not _config_matches_template(generated, template):
            raise SystemExit(
                "ERROR: QAT regional module names do not match the current prepared training graph"
            )
        print("QAT config FX node structure: PASS")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")
    report = {
        "task": task,
        "model_config": str(Path(args.model_config).resolve()),
        "weights": str(Path(args.weights).resolve()) if args.weights else None,
        "base_config": str(args.base_config.resolve()),
        "output": str(args.output.resolve()) if args.output else None,
        "image_shape": [args.batch_size, *args.image_shape],
        "reparameterized": args.reparameterize,
        "attention_dtype_requested": args.attention_dtype,
        "attention_dtype": resolved_attention_dtype,
        **discovery,
    }
    if args.report is not None:
        if args.report.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for position, region in enumerate(discovery["attention"], start=1):
        print(f"attention {position}: {region['owner']}")
        print(
            "  "
            f"qkv={region['qkv_linear']} proj={region['proj_linear']} "
            f"scale={region['scale_mul']} matmul={region['first_matmul']} "
            f"softmax={region['softmax']} matmul_out={region['second_matmul']}"
        )
    if not discovery["attention"]:
        print("attention regions: 0 (global QAT config only)")
    else:
        print(f"attention dtype: {resolved_attention_dtype}")
    if args.output is not None:
        print(f"generated config: {args.output}")


if __name__ == "__main__":
    main()
