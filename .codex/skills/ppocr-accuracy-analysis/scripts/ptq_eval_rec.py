"""PTQ accuracy validation for reparameterized PP-OCR recognition models.

Evaluates the LSQ pipeline WITHOUT training: float weights -> reparameterized
deployment (CTC) graph -> LSQ prepare -> static weight statistics + activation
statistics over calibration batches from the train set -> validation metrics
for both the prepared fake-quant-on and the converted PT2E graph.

Default width sweep uses the full-global configs (base_u8s8 / base_u16s16,
no attention regional entries) with the LSQ quantizer, mirroring the v5 mobile
rec PTQ study recorded in
docs/references/development/axera_qat/records/icdar2015_ppocrv5_mobile_rec_qat_training.md §43.3.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.ao.quantization import (
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    enable_observer,
    move_exported_model_to_eval,
)
from torch.utils.data import DataLoader, Subset

from pytorchocr.quantization import (
    build_qat_dynamic_shapes,
    convert_prepared_model,
    initialize_weight_observers,
    load_axera_quantizer,
    prepare_qat_model,
)
from pytorchocr.training import (
    build_dataset,
    build_task_model,
    build_validation_metric,
    load_ocr_config,
)

DEFAULT_IMAGE_SHAPE = (3, 48, 320)


def build_loader(config, model_config, labels, data_dir, image_shape, batch_size, subset):
    dataset = build_dataset(
        "rec",
        model_config,
        config,
        image_shape,
        labels,
        data_dir,
        return_polygons=False,
        rec_multi_head=False,
    )
    if subset:
        dataset = Subset(dataset, range(subset))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False
    )


def evaluate(model, loader, metric, device):
    metric.reset()
    output_finite = True
    start = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = {
                key: (value.to(device) if torch.is_tensor(value) else value)
                for key, value in targets.items()
            }
            output = model(images)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output_finite &= bool(torch.isfinite(output).all())
            metric.update(output.detach().cpu(), targets)
    elapsed = time.perf_counter() - start
    return {
        "output_finite": output_finite,
        "elapsed_seconds": elapsed,
        **metric.compute(),
    }


def run_ptq(
    qat_config,
    model_config,
    weights,
    train_labels,
    val_labels,
    data_dir,
    image_shape,
    device,
    calibration_batches,
    calibration_batch_size,
):
    config = load_ocr_config(model_config)
    model = build_task_model(
        "rec",
        model_config,
        weights_path=weights,
        reparameterize=True,
        rec_graph="deploy",
    )
    images = torch.zeros([calibration_batch_size, *image_shape], dtype=torch.float32)
    quantizer = load_axera_quantizer(qat_config)
    prepared, _ = prepare_qat_model(
        model,
        (images,),
        quantizer,
        dynamic_shapes=build_qat_dynamic_shapes(
            images,
            dynamic_batch=True,
            dynamic_heights=None,
            max_batch=calibration_batch_size,
        ),
    )
    del model
    initialize_weight_observers(prepared)
    prepared = prepared.to(device)
    move_exported_model_to_eval(prepared)
    train_loader = build_loader(
        config,
        model_config,
        train_labels,
        data_dir,
        image_shape,
        calibration_batch_size,
        calibration_batches * calibration_batch_size,
    )
    prepared.apply(disable_fake_quant)
    prepared.apply(enable_observer)
    with torch.no_grad():
        for batch_index, (images, _targets) in enumerate(train_loader):
            prepared(images.to(device))
            if batch_index + 1 >= calibration_batches:
                break
    prepared.apply(enable_fake_quant)
    prepared.apply(disable_observer)

    val_loader = build_loader(
        config, model_config, val_labels, data_dir, image_shape, calibration_batch_size, None
    )
    metric = build_validation_metric("rec", config, config_path=model_config)
    prepared_result = evaluate(prepared, val_loader, metric, device)

    converted = convert_prepared_model(prepared)
    del prepared
    metric = build_validation_metric("rec", config, config_path=model_config)
    converted_result = evaluate(converted, val_loader, metric, device)
    return {
        "qat_config": qat_config,
        "calibration_batches": calibration_batches,
        "calibration_batch_size": calibration_batch_size,
        "prepared_fake_quant_on": prepared_result,
        "converted": converted_result,
    }


def resolve_configs(bitwidths, custom_configs):
    payloads = []
    for bitwidth in bitwidths:
        if bitwidth == 8:
            base = ROOT_DIR / "configs/qat/base_u8s8.json"
            label = "full_u8s8"
        elif bitwidth == 16:
            base = ROOT_DIR / "configs/qat/base_u16s16.json"
            label = "full_u16s16"
        else:
            raise ValueError(f"Unsupported bitwidth: {bitwidth} (use 8 or 16)")
        config = json.loads(base.read_text(encoding="utf-8"))
        config["lsq"] = True
        path = f"/tmp/opencode/ptq_{label}_lsq.json"
        Path(path).write_text(json.dumps(config), encoding="utf-8")
        payloads.append((label, path))
    for index, custom in enumerate(custom_configs):
        payloads.append((f"custom_{index}", custom))
    return payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--train-label-file", required=True)
    parser.add_argument("--val-label-file", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--image-shape", nargs=3, type=int, default=list(DEFAULT_IMAGE_SHAPE))
    parser.add_argument("--bitwidths", nargs="+", type=int, choices=(8, 16), default=None)
    parser.add_argument("--qat-config", nargs="+", default=None,
                        help="Custom QAT JSON path(s); evaluated in addition to --bitwidths.")
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--calibration-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.bitwidths and not args.qat_config:
        parser.error("Provide --bitwidths and/or --qat-config.")

    results = {}
    for label, qat_config in resolve_configs(args.bitwidths or [], args.qat_config or []):
        results[label] = run_ptq(
            qat_config,
            args.model_config,
            args.weights,
            args.train_label_file,
            args.val_label_file,
            args.data_dir,
            tuple(args.image_shape),
            args.device,
            args.calibration_batches,
            args.calibration_batch_size,
        )
        print(json.dumps({label: results[label]}, indent=2), flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
