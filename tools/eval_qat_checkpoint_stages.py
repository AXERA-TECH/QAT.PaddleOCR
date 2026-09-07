"""Evaluate a QAT checkpoint: three-state accuracy + re-calibration.

States (all evaluated on the full --label-file set):
  trained_fake_off    trained weights, fake quant disabled  (float domain)
  trained_fake_on     trained weights, fake quant enabled   (training observer stats)
  trained_converted   convert of trained_fake_on
Re-calibration: re-run observers on independent training-set samples (default 512),
then evaluate fake_on / converted again.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.ao.quantization import (
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    enable_observer,
    move_exported_model_to_eval,
)
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pytorchocr.diagnostics import (
    build_prepared_qat_checkpoint,
    load_qat_checkpoint,
    observer_qparams,
)
from pytorchocr.quantization import convert_prepared_model
from pytorchocr.training import (
    build_dataset,
    build_validation_metric,
    load_ocr_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--calibration-label-file", required=True)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(20260819)
    checkpoint, metadata = load_qat_checkpoint(args.checkpoint)
    config = load_ocr_config(metadata["model_config"])
    max_text_length = int(config["Global"].get("max_text_length", 25))
    prepared, _ = build_prepared_qat_checkpoint(metadata)
    prepared.load_state_dict(checkpoint["model"], strict=True)
    prepared.apply(disable_observer)
    move_exported_model_to_eval(prepared)

    dataset = build_dataset(
        "rec",
        metadata["model_config"],
        config,
        tuple(metadata["image_shape"]),
        args.label_file,
        args.data_dir,
    )
    device = torch.device(args.device)

    def evaluate(model, name, results):
        model.to(device)
        metric = build_validation_metric("rec", config, config_path=metadata["model_config"])
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )
        finite = True
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device, non_blocking=True)
                gtc = torch.zeros(images.shape[0], max_text_length, dtype=torch.long,
                                  device=device)
                out = model(images, gtc)
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out["ctc"]
                finite &= bool(torch.isfinite(logits).all())
                metric.update(logits, targets)
        model.cpu()
        results[name] = {**metric.compute(), "finite": finite}
        print(f"{name}: {json.dumps(results[name])}", flush=True)

    results = {}
    trained_fake_off = copy.deepcopy(prepared)
    trained_fake_off.apply(disable_fake_quant)
    evaluate(trained_fake_off, "trained_fake_off", results)
    del trained_fake_off

    trained_fake_on = prepared
    trained_fake_on.apply(enable_fake_quant)
    evaluate(trained_fake_on, "trained_fake_on", results)
    converted = convert_prepared_model(copy.deepcopy(trained_fake_on))
    evaluate(converted, "trained_converted", results)

    # Re-calibration on independent training-set samples.
    prepared.apply(enable_observer)
    calib_dataset = build_dataset(
        "rec",
        metadata["model_config"],
        config,
        tuple(metadata["image_shape"]),
        args.calibration_label_file,
        args.data_dir,
    )
    calib_loader = DataLoader(
        Subset(calib_dataset, range(min(args.calibration_samples, len(calib_dataset)))),
        batch_size=args.calibration_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    prepared.to(device)
    with torch.no_grad():
        for images, _ in calib_loader:
            images = images.to(device, non_blocking=True)
            gtc = torch.zeros(images.shape[0], max_text_length, dtype=torch.long,
                              device=device)
            out = prepared(images, gtc)
            logits = out[0] if isinstance(out, tuple) else out["ctc"]
            if not torch.isfinite(logits).all():
                raise RuntimeError("Re-calibration produced non-finite output.")
    prepared.cpu()
    prepared.apply(disable_observer)
    move_exported_model_to_eval(prepared)
    evaluate(prepared, "recalibrated_fake_on", results)
    recal_converted = convert_prepared_model(copy.deepcopy(prepared))
    evaluate(recal_converted, "recalibrated_converted", results)

    results["calibration_samples"] = args.calibration_samples
    results["qparams"] = observer_qparams(prepared)
    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
