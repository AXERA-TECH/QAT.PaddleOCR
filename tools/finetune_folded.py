#!/usr/bin/env python3
"""Fold a non-reparameterized QAT checkpoint into a single-branch model and
finetune it (quantized-domain folding workflow).

Normalized CLI for the workflow previously living in the
ppocr-quantized-domain-fold skill scripts. Reusable logic lives in
pytorchocr.quantization.folding.

Two modes:

1. Extract folded state and eval (no training):
     --source-checkpoint runs/expN_.../best.pt --fold-state-output /tmp/folded.pt
   With --qat-config, runs the quantized-domain (prepare+convert) evaluation
   which is the real finetune starting point. Without it, only a raw float
   forward is reported (NOT comparable to training accuracy).

2. Finetune the folded model:
     --source-checkpoint ... --folded-state /tmp/folded.pt --output-dir runs/...
   Saves epoch_NNNN.pt, best.pt (with epoch/metric metadata), last.pt.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.ao.quantization import (
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    enable_observer,
    move_exported_model_to_eval,
    move_exported_model_to_train,
)
from torch.utils.data import DataLoader

from pytorchocr.quantization import (
    FullRecTrainingWrapper,
    build_folded_state,
    build_qat_dynamic_shapes,
    convert_prepared_model,
    enable_learn,
    folded_eager_model,
    initialize_weight_observers,
    load_axera_quantizer,
    prepare_qat_model,
    transfer_activation_qparams_by_site,
)
from pytorchocr.training import (
    Trainer,
    build_criterion,
    build_dataset,
    build_optimizer,
    build_task_model,
    build_validation_metric,
    load_ocr_config,
    load_training_profile,
    profile_value,
    training_log,
    update_best_validation,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="rec", choices=["rec"])
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--label-file")
    parser.add_argument("--data-dir")
    parser.add_argument("--val-label-file")
    parser.add_argument("--val-data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--training-profile")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--source-checkpoint",
        required=True,
        help="Non-reparameterized QAT checkpoint (best.pt).",
    )
    parser.add_argument(
        "--folded-state",
        help="Precomputed folded state (.pt); if absent it is extracted first.",
    )
    parser.add_argument(
        "--fold-state-output",
        help="Where to write the extracted folded state (default: next to "
        "source checkpoint as <name>_folded_state.pt).",
    )
    parser.add_argument(
        "--qat-config",
        help="QAT JSON for prepare; defaults to the training profile qat_config.",
    )
    parser.add_argument("--smoke-steps", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument(
        "--activation-qparam-transfer",
        choices=("none", "semantic"),
        default="none",
        help=(
            "Optional activation qparam warm-start from the source prepared "
            "graph. Default 'none' re-observes the folded graph. 'semantic' "
            "copies only scale/zero_point for matching FX producer sites."
        ),
    )
    parser.add_argument(
        "--activation-qparam-include-static",
        action="store_true",
        help=(
            "Also consider get_attr/static-constant observer sites during "
            "semantic activation qparam transfer. Disabled by default."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir:
        with training_log(args.output_dir):
            return _main(args)
    return _main(args)


def _main(args):
    config = load_ocr_config(args.model_config)
    profile = load_training_profile(args.training_profile)
    if profile is not None:
        profile.validate_contract(args.task, config["Global"].get("model_name"))

    torch.manual_seed(profile_value(args.finetune_epochs, profile, "seed", 20260813))
    image_shape = tuple(profile.training.get("image_shape", [3, 48, 320]))
    batch_size = int(profile.training.get("batch_size", 64))
    workers = int(profile.training.get("workers", 8))
    qat_config = str(args.qat_config or profile.qat_config)
    max_text_length = int(config["Global"].get("max_text_length", 25))

    # 1) Extract folded state if not provided
    if args.folded_state:
        folded_state = torch.load(args.folded_state, map_location="cpu", weights_only=False)
    else:
        ckpt = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
        folded_state = build_folded_state(ckpt, args.model_config, args.weights)
        output = args.fold_state_output or (
            Path(args.source_checkpoint).with_name(
                f"{Path(args.source_checkpoint).stem}_folded_state.pt"
            )
        )
        torch.save(folded_state, output)
        print("saved folded state:", output)

    # 2) Folded eager model (finetune start point)
    model, applied, skipped = folded_eager_model(
        args.source_checkpoint, args.model_config, args.weights, folded_state
    )
    print(f"folded state applied: {applied} | skipped: {skipped}")

    # 3) Optional standalone quantized-domain eval (no training)
    if not args.label_file and not args.output_dir:
        _quantized_domain_eval(args, config, model, image_shape, batch_size,
                               workers, qat_config, max_text_length)
        return

    if not args.label_file or not args.data_dir or not args.val_label_file:
        raise ValueError(
            "Finetune requires --label-file/--data-dir/--val-label-file/"
            "--val-data-dir/--output-dir."
        )

    # 4) Data loaders
    dataset = build_dataset(
        "rec", args.model_config, config, image_shape,
        args.label_file, args.data_dir,
        rec_multi_head=True, augmentation="none",
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, drop_last=True,
    )
    val_dataset = build_dataset(
        "rec", args.model_config, config, image_shape,
        args.val_label_file, args.val_data_dir,
        rec_multi_head=True, augmentation="none",
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, drop_last=False)

    # 5) Prepare + LSQ (observer statistics pass on real training data)
    images, targets = next(iter(loader))
    wrapper = FullRecTrainingWrapper(
        model, max_text_length=max_text_length
    ).set_qat_capture_mode()
    prepared, float_nodes = prepare_qat_model(
        wrapper,
        (images, targets["gtc_targets"]),
        load_axera_quantizer(qat_config),
        dynamic_shapes=build_qat_dynamic_shapes(
            images, dynamic_batch=True, dynamic_heights=None,
            batch_aligned_inputs=1, max_batch=batch_size,
        ),
    )
    prepared.graph_role = "pretrained_train"
    prepared.model_type = "rec"
    prepared.output_names = FullRecTrainingWrapper.output_names
    move_exported_model_to_train(prepared)
    lsq = bool(profile.training.get("lsq", True)) if profile else True
    if lsq:
        initialize_weight_observers(prepared)
        prepared.apply(enable_learn)
        prepared.apply(disable_fake_quant)
        prepared.apply(enable_observer)
        with torch.no_grad():
            prepared(images, targets["gtc_targets"])
        prepared.apply(enable_fake_quant)
        prepared.apply(disable_observer)
    else:
        initialize_weight_observers(prepared)
        prepared.apply(enable_observer)
        with torch.no_grad():
            prepared(images, targets["gtc_targets"])
        prepared.apply(disable_observer)
    _maybe_transfer_activation_qparams(prepared, args)
    print("prepared, float nodes:", float_nodes)

    # 6) Finetune with smoke gate
    criterion = build_criterion("rec", config, rec_multi_head=True)
    optimizer = build_optimizer(
        prepared, config,
        learning_rate=float(profile.training.get("learning_rate", 1e-5)),
        weight_decay=float(profile.training.get("weight_decay", 1e-4)),
        lab_lr_multiplier=0.1, ctc_fc_weight_decay=1e-5,
        optimizer_name="AdamW",
    )
    metric = build_validation_metric("rec", config, config_path=args.model_config)
    trainer = Trainer(prepared, criterion, optimizer, device=args.device, amp=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.finetune_epochs or int(profile.training.get("epochs", 20))

    print(f"=== smoke gate: {args.smoke_steps} steps ===")
    for step in range(1, args.smoke_steps + 1):
        smoke_images, smoke_targets = next(iter(loader))
        smoke_losses = trainer.train_step(smoke_images, smoke_targets)
        if not torch.isfinite(torch.tensor(smoke_losses["loss"])):
            raise RuntimeError(
                f"Smoke gate failed at step {step}: non-finite loss."
            )
        if step in (1, 3, args.smoke_steps):
            print(f"smoke step {step}: loss={smoke_losses['loss']:.4f}")
    print("smoke gate passed")

    def _metadata():
        return {
            "task": "rec",
            "qat": True,
            "model_config": str(Path(args.model_config).resolve()),
            "qat_config": str(Path(qat_config).resolve()),
            "weights": str(Path(args.weights).resolve()),
            "reparameterized": True,
            "rec_graph": "pretrained_train",
            "image_shape": list(image_shape),
            "batch_size": batch_size,
            "folded_finetune": True,
            "source": str(Path(args.source_checkpoint).resolve()),
            "folded_state": str(args.folded_state or Path(output).resolve()),
        }

    main_indicator = str(metric.main_indicator)
    best_epoch = 0
    for epoch in range(epochs):
        loss_sums = {}
        loss_counts = {}
        for images, targets in loader:
            step_losses = trainer.train_step(images, targets)
            for name, value in step_losses.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + value
                loss_counts[name] = loss_counts.get(name, 0) + 1
        train_loss = {
            name: loss_sums[name] / loss_counts[name]
            for name in loss_sums
        }
        results = trainer.evaluate(val_loader, metric=metric)
        print(json.dumps({
            "epoch": epoch + 1,
            "steps": len(loader),
            **{f"train_{k}": round(float(v), 6) for k, v in train_loss.items()},
            **{f"val_{k}": round(float(v), 6) if isinstance(v, float) else v
               for k, v in results.items()},
        }, sort_keys=True), flush=True)
        trainer.save_checkpoint(
            output_dir / f"epoch_{epoch + 1:04d}.pt",
            metadata=_metadata(),
        )
        is_best = update_best_validation(trainer, results, metric=metric)
        if is_best:
            best_epoch = epoch + 1
            trainer.epoch = best_epoch
            trainer.save_checkpoint(output_dir / "best.pt", metadata=_metadata())
            print(
                f"new best at epoch {best_epoch}: "
                f"{main_indicator}={trainer.best_validation_metric_value:.6f}"
            )

    trainer.epoch = epochs
    trainer.save_checkpoint(output_dir / "last.pt", metadata=_metadata())
    print(
        f"finetune done (best {main_indicator}="
        f"{trainer.best_validation_metric_value:.6f} at epoch {best_epoch})"
    )


def _quantized_domain_eval(args, config, model, image_shape, batch_size, workers,
                           qat_config, max_text_length):
    """prepare + convert + validation accuracy (real finetune starting point)."""
    from pytorchocr.quantization import (
        build_qat_dynamic_shapes,
        convert_prepared_model,
        load_axera_quantizer,
        prepare_qat_model,
    )

    images = torch.randn(batch_size, *image_shape)
    gtc = torch.zeros(batch_size, max_text_length, dtype=torch.int64)
    wrapper = FullRecTrainingWrapper(
        model, max_text_length=max_text_length
    ).set_qat_capture_mode()
    prepared, _ = prepare_qat_model(
        wrapper, (images, gtc),
        load_axera_quantizer(qat_config),
        dynamic_shapes=build_qat_dynamic_shapes(
            images, dynamic_batch=True, dynamic_heights=None,
            batch_aligned_inputs=1, max_batch=batch_size,
        ),
    )
    prepared.graph_role = "pretrained_train"
    prepared.model_type = "rec"
    prepared.output_names = FullRecTrainingWrapper.output_names
    move_exported_model_to_train(prepared)
    initialize_weight_observers(prepared)
    prepared.apply(enable_learn)
    prepared.apply(disable_fake_quant)
    prepared.apply(enable_observer)
    with torch.no_grad():
        prepared(images, gtc)
    prepared.apply(enable_fake_quant)
    prepared.apply(disable_observer)
    _maybe_transfer_activation_qparams(prepared, args)

    move_exported_model_to_eval(prepared)
    converted = convert_prepared_model(prepared)
    move_exported_model_to_eval(converted)
    converted.to(args.device)

    val_dataset = build_dataset(
        "rec", args.model_config, config, image_shape,
        args.val_label_file, args.val_data_dir,
        rec_multi_head=True, augmentation="none",
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, drop_last=False)
    metric = build_validation_metric("rec", config, config_path=args.model_config)
    metric.reset()
    with torch.no_grad():
        for v_images, v_targets in val_loader:
            v_images = v_images.to(args.device)
            out = converted(v_images, v_targets["gtc_targets"].to(args.device))
            out = out[0] if isinstance(out, tuple) else out["ctc"]
            metric.update(out.cpu(), v_targets)
    print(
        "quantized-domain folded eval (finetune-start accuracy):",
        json.dumps(metric.compute(), sort_keys=True),
    )


def _maybe_transfer_activation_qparams(prepared, args):
    if args.activation_qparam_transfer == "none":
        return None
    from pytorchocr.diagnostics import (
        build_prepared_qat_checkpoint,
        load_qat_checkpoint,
    )

    source_checkpoint, source_metadata = load_qat_checkpoint(args.source_checkpoint)
    source_prepared, _ = build_prepared_qat_checkpoint(source_metadata)
    new_state, report = transfer_activation_qparams_by_site(
        source_prepared,
        source_checkpoint["model"],
        prepared,
        include_get_attr=bool(args.activation_qparam_include_static),
    )
    prepared.load_state_dict(new_state, strict=True)
    print(
        "activation qparam transfer:",
        json.dumps(report.as_dict(), sort_keys=True),
    )
    return report


if __name__ == "__main__":
    main()
