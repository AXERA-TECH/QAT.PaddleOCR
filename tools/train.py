import argparse
import faulthandler
import json
import random
import signal
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pytorchocr.quantization import (
    FullRecTrainingWrapper,
    build_qat_dynamic_shapes,
    convert_prepared_model,
    disable_learn,
    enable_learn,
    initialize_kept_bn_statistics,
    initialize_weight_observers,
    is_lsq_config,
    load_axera_quantizer,
    prepare_qat_model,
)
from torch.ao.quantization import (
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    enable_observer,
    move_exported_model_to_eval,
    move_exported_model_to_train,
)
from pytorchocr.training import (
    RecognitionMultiScaleBatchSampler,
    Trainer,
    build_criterion,
    build_dataset,
    build_kd_criterion,
    build_optimizer,
    build_scheduler,
    build_task_model,
    build_validation_metric,
    config_value,
    default_kd_layers,
    detection_collate,
    epoch2_accuracy_guard,
    file_sha256,
    load_ocr_config,
    load_training_profile,
    profile_value,
    update_best_validation,
    validate_resume_contract,
)
from pytorchocr.training.profile import copy_run_configs


def _round_summary_floats(summary):
    """Round every float in the summary dict to 6 decimal places for logging."""
    return {
        key: (
            round(value, 6)
            if isinstance(value, float)
            else _round_summary_floats(value)
            if isinstance(value, dict)
            else value
        )
        for key, value in summary.items()
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PaddleOCR PyTorch float or Axera QAT models."
    )
    parser.add_argument("--task", choices=["det", "rec"], required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weights", help="Converted PyTorch pretrained weights.")
    parser.add_argument("--resume", help="Trainer checkpoint to resume.")
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--val-label-file")
    parser.add_argument(
        "--val-data-dir",
        help="Validation data root; defaults to --data-dir.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--training-profile",
        help="QAT training defaults; explicit CLI values take precedence.",
    )
    parser.add_argument(
        "--qat",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--qat-config")
    parser.add_argument(
        "--axera-root",
        help="Deprecated compatibility option; the quantizer is vendored.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--image-shape", nargs=3, type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--observer-freeze-epoch", type=int)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--lr-final-factor", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--rec-ctc-backbone-grad",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--rec-graph",
        choices=["deploy", "pretrained_train"],
        default=None,
        help="Recognition graph role; full pretrained training keeps CTC+NRTR.",
    )
    parser.add_argument("--lab-lr-multiplier", type=float)
    parser.add_argument("--ctc-fc-weight-decay", type=float)
    parser.add_argument(
        "--optimizer",
        choices=["Adam", "AdamW", "SGD", "Momentum"],
        help="Override the optimizer declared by the Paddle model YAML.",
    )
    parser.add_argument("--momentum", type=float)
    parser.add_argument(
        "--dynamic-heights",
        nargs="+",
        type=int,
        help="Allowed recognition heights for the prepared PT2E training graph.",
    )
    parser.add_argument(
        "--augmentation",
        choices=["none", "paddle"],
        help="Training augmentation preset; validation is always deterministic.",
    )
    parser.add_argument(
        "--multi-scale-training",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Sample recognition training batches at --dynamic-heights.",
    )
    parser.add_argument("--save-every", type=int)
    parser.add_argument(
        "--float-accuracy-baseline",
        type=float,
        help="Recognition float accuracy used by the epoch-2 QAT guard.",
    )
    parser.add_argument(
        "--epoch2-max-accuracy-drop",
        type=float,
        help="Stop after epoch 2 when validation acc drops by this amount.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load --resume and run one validation pass without training.",
    )
    parser.add_argument(
        "--eval-stage",
        choices=["float", "prepared", "converted"],
        default="prepared",
        help="PT2E graph stage used by --eval-only (default: prepared).",
    )
    parser.add_argument(
        "--reparameterize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: enabled for QAT and disabled for float training.",
    )
    parser.add_argument(
        "--insert-identity-bn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Insert identity BatchNorm2d after every fused conv before PT2E "
            "QAT prepare (fusion pass is kept enabled)."
        ),
    )
    parser.add_argument(
        "--bn-statistics-steps",
        type=int,
        default=None,
        help=(
            "Batches used to measure fused-conv statistics for kept identity "
            "BNs (QARepVGG insert_bn style) before QAT prepare."
        ),
    )
    parser.add_argument(
        "--bn-training-momentum",
        type=float,
        default=None,
        help=(
            "BatchNorm running-stats momentum used during QAT training for "
            "kept identity BNs; larger values (e.g. 0.9) keep running_var "
            "fresh so gamma/sqrt(var+eps) stays bounded at eval time."
        ),
    )
    parser.add_argument(
        "--freeze-bn-stats",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Freeze kept-BN running statistics during QAT training "
            "(momentum=0.0): locks the folding denominator "
            "gamma/sqrt(var+eps) to its initialized value and prevents BN "
            "folding explosion; gamma/beta stay trainable."
        ),
    )
    parser.add_argument(
        "--lsq",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use the learnable-step-size (LSQ) quantizer; requires a QAT JSON "
            "with top-level 'lsq': true."
        ),
    )

    parser.add_argument(
        "--kd",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable knowledge distillation with a frozen float teacher.",
    )
    parser.add_argument(
        "--teacher-weights",
        help="Teacher weights; defaults to the student float --weights.",
    )
    parser.add_argument(
        "--teacher-model-config",
        help="Teacher model YAML; defaults to --model-config.",
    )
    parser.add_argument(
        "--kd-mode",
        choices=["logits", "logits_mse", "maps"],
        help="KD loss mode for the task head (rec: logits/logits_mse, det: maps).",
    )
    parser.add_argument("--kd-weight", type=float)
    parser.add_argument("--kd-temperature", type=float)
    parser.add_argument(
        "--kd-neck-weight",
        type=float,
        help="Intermediate neck feature KD weight (default 0, disabled).",
    )
    parser.add_argument(
        "--kd-backbone-weight",
        type=float,
        help="Intermediate backbone feature KD weight (default 0, disabled).",
    )
    return parser.parse_args()


def _set_intermediate_exposure(model, task, qat, rec_graph):
    """Expose intermediate features for multi-layer KD on the student side.

    rec QAT uses FullRecTrainingWrapper(expose_intermediates=...) at
    construction; rec float training consumes the MultiHead top-level dict and
    must NOT enable return_all_feats (MultiLoss requires the ctc/gtc keys at
    the top level). det uses DetTrainingWrapper (QAT) or a bare model with
    return_all_feats=True (float).
    """
    if task == "det":
        if qat:
            model.expose_intermediates = True
        else:
            model.return_all_feats = True


def build_teacher(args, qat, reparameterize, rec_graph, rec_ctc_backbone_grad):
    teacher_config = args.teacher_model_config or args.model_config
    teacher_weights = args.teacher_weights or args.weights
    teacher = build_task_model(
        args.task,
        teacher_config,
        weights_path=teacher_weights,
        reparameterize=reparameterize,
        det_graph="training" if qat else "pretrained_train",
        rec_ctc_backbone_grad=rec_ctc_backbone_grad,
        rec_graph=rec_graph,
    )
    if args.task == "rec":
        # The frozen teacher must emit full training-branch outputs (including
        # backbone/neck features) for multi-layer KD; _freeze_teacher keeps its
        # BatchNorm layers in eval mode.
        if rec_graph == "pretrained_train":
            teacher.return_all_feats = True
    else:
        _set_intermediate_exposure(teacher, args.task, qat, rec_graph)
    return teacher


def build_kd_layers(args, kd_mode):
    layers = default_kd_layers(args.task, kd_mode=kd_mode, head_weight=1.0)
    if args.kd_neck_weight:
        neck_key = "ctc_neck" if args.task == "rec" else "neck_out"
        layers[neck_key] = ("mse", float(args.kd_neck_weight))
    if args.kd_backbone_weight:
        layers["backbone_out"] = ("mse", float(args.kd_backbone_weight))
    return layers


def _prepare_lsq_training(model, loader, rec_multi_head, resume=False):
    """Initialize LSQ scales and enable learnable-step training.

    Follows QAT.Ultralytics.YOLOv5 qat_base_ptq.py:
    1. static weight observer pass (existing initialize_weight_observers)
       fills per-channel weight scales while _LearnableFakeQuantize modules are
       still in static mode;
    2. enable_param_learning() on every _LearnableFakeQuantize module;
    3. one statistics forward (fake quant off, observer on) over the first
       loader batch to initialize activation scales, then switch fake quant
       back on and observers off.

    When resuming, scales already carry trained values from the checkpoint:
    only re-enable learning and skip both static initialization passes.
    """
    if resume:
        model.apply(enable_learn)
        return
    # Static weight-parameter statistics FIRST while every module is still in
    # static mode (enable_learn would disable observer updates and leave the
    # weight scales at their default value, collapsing quantized weights).
    initialize_weight_observers(model)
    model.apply(enable_learn)
    images, targets = next(iter(loader))
    images = images.detach()
    targets = _detach_targets(targets)
    model.apply(disable_fake_quant)
    model.apply(enable_observer)
    try:
        with torch.no_grad():
            if rec_multi_head:
                model(images, targets["gtc_targets"])
            else:
                model(images)
    finally:
        model.apply(enable_fake_quant)
        model.apply(disable_observer)


def _detach_targets(targets):
    if torch.is_tensor(targets):
        return targets.detach()
    if isinstance(targets, dict):
        return {key: _detach_targets(value) for key, value in targets.items()}
    if isinstance(targets, (tuple, list)):
        return [_detach_targets(value) for value in targets]
    return targets


def train(args):
    if args.eval_only and not args.val_label_file:
        raise ValueError("--eval-only requires --val-label-file.")
    if not args.eval_only and args.eval_stage != "prepared":
        raise ValueError("--eval-stage is only valid with --eval-only.")
    config = load_ocr_config(args.model_config)
    global_config = config.get("Global", {})
    profile = load_training_profile(args.training_profile)
    if profile is not None:
        profile.validate_contract(args.task, global_config.get("model_name"))
    qat = bool(config_value(args.qat, profile.qat if profile is not None else False))
    seed = profile_value(args.seed, profile, "seed")
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if args.eval_only:
        if args.eval_stage == "float" and qat:
            raise ValueError("--eval-stage float requires --no-qat.")
        if args.eval_stage != "float" and not qat:
            raise ValueError("Prepared/converted evaluation requires QAT.")
        if args.eval_stage != "float" and not args.resume:
            raise ValueError("Prepared/converted evaluation requires --resume.")
    qat_config = config_value(
        args.qat_config,
        str(profile.qat_config) if profile is not None and profile.qat_config else None,
    )
    train_config = config.get("Train", {})
    loader_config = train_config.get("loader", {})
    epochs = int(
        profile_value(
            args.epochs,
            profile,
            "epochs",
            global_config.get("epoch_num", 1),
        )
    )
    batch_size = int(
        profile_value(
            args.batch_size,
            profile,
            "batch_size",
            loader_config.get("batch_size_per_card", 1),
        )
    )
    workers = int(
        profile_value(
            args.workers,
            profile,
            "workers",
            loader_config.get("num_workers", 0),
        )
    )
    image_shape = tuple(
        profile_value(
            args.image_shape,
            profile,
            "image_shape",
            global_config.get("d2s_train_image_shape"),
        )
    )
    if len(image_shape) != 3:
        raise ValueError("Training image shape must be [channels, height, width].")
    if qat and not qat_config:
        raise ValueError("QAT training requires --qat-config.")
    save_every = int(profile_value(args.save_every, profile, "save_every", 1))
    if save_every <= 0:
        raise ValueError("--save-every must be positive.")
    observer_freeze_epoch = profile_value(
        args.observer_freeze_epoch,
        profile,
        "observer_freeze_epoch",
    )
    rec_ctc_backbone_grad = bool(
        profile_value(
            args.rec_ctc_backbone_grad,
            profile,
            "rec_ctc_backbone_grad",
            False,
        )
    )
    default_rec_graph = "pretrained_train" if qat and args.task == "rec" else "deploy"
    rec_graph = profile_value(args.rec_graph, profile, "rec_graph", default_rec_graph)
    float_accuracy_baseline = profile_value(
        args.float_accuracy_baseline,
        profile,
        "float_accuracy_baseline",
    )
    epoch2_max_accuracy_drop = profile_value(
        args.epoch2_max_accuracy_drop,
        profile,
        "epoch2_max_accuracy_drop",
    )
    if (float_accuracy_baseline is None) != (epoch2_max_accuracy_drop is None):
        raise ValueError(
            "--float-accuracy-baseline and --epoch2-max-accuracy-drop "
            "must be configured together."
        )
    if float_accuracy_baseline is not None:
        if not qat or args.task != "rec":
            raise ValueError("The epoch-2 accuracy guard is only valid for rec QAT.")
        if not args.val_label_file:
            raise ValueError("The epoch-2 accuracy guard requires validation data.")
        if not 0 <= float(float_accuracy_baseline) <= 1:
            raise ValueError("--float-accuracy-baseline must be in [0, 1].")
        if not 0 <= float(epoch2_max_accuracy_drop) <= 1:
            raise ValueError("--epoch2-max-accuracy-drop must be in [0, 1].")
    augmentation = str(
        profile_value(args.augmentation, profile, "augmentation", "none")
    )
    multi_scale_training = bool(
        profile_value(
            args.multi_scale_training,
            profile,
            "multi_scale_training",
            False,
        )
    )
    dynamic_heights = profile_value(
        args.dynamic_heights,
        profile,
        "dynamic_heights",
    )
    if dynamic_heights is not None:
        dynamic_heights = list(dynamic_heights)
        if args.task != "rec":
            raise ValueError("--dynamic-heights is only valid for recognition.")
        if not qat and not multi_scale_training:
            raise ValueError(
                "Float --dynamic-heights requires --multi-scale-training."
            )
        if image_shape[1] not in dynamic_heights:
            raise ValueError(
                "The training image height must be included in --dynamic-heights."
            )
    if rec_ctc_backbone_grad and args.task != "rec":
        raise ValueError("--rec-ctc-backbone-grad is only valid for recognition.")
    rec_multi_head = args.task == "rec" and rec_graph == "pretrained_train"
    if args.task != "rec" and rec_graph != "deploy":
        raise ValueError("--rec-graph is only valid for recognition.")
    if multi_scale_training and args.task != "rec":
        raise ValueError("--multi-scale-training is only valid for recognition.")
    if multi_scale_training and not dynamic_heights:
        raise ValueError("--multi-scale-training requires --dynamic-heights.")

    train_generator = None
    if seed is not None:
        train_generator = torch.Generator().manual_seed(seed)

    dataset = build_dataset(
        args.task,
        args.model_config,
        config,
        image_shape,
        args.label_file,
        args.data_dir,
        rec_multi_head=rec_multi_head,
        augmentation="none" if args.eval_only else augmentation,
    )
    if multi_scale_training:
        batch_sampler = RecognitionMultiScaleBatchSampler(
            dataset,
            width=image_shape[2],
            heights=dynamic_heights,
            base_height=image_shape[1],
            base_batch_size=batch_size,
            fix_batch_size=False,
            drop_last=qat or bool(loader_config.get("drop_last", False)),
            seed=seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=workers,
            pin_memory=str(args.device).startswith("cuda"),
            generator=train_generator,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=bool(loader_config.get("shuffle", True)),
            num_workers=workers,
            drop_last=qat or bool(loader_config.get("drop_last", False)),
            pin_memory=str(args.device).startswith("cuda"),
            generator=train_generator,
        )
    validation_loader = None
    if args.val_label_file:
        validation_dataset = build_dataset(
            args.task,
            args.model_config,
            config,
            image_shape,
            args.val_label_file,
            args.val_data_dir or args.data_dir,
            return_polygons=args.task == "det",
            rec_multi_head=rec_multi_head,
            augmentation="none",
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            drop_last=False,
            pin_memory=str(args.device).startswith("cuda"),
            collate_fn=detection_collate if args.task == "det" else None,
        )
    if len(loader) == 0:
        raise ValueError(
            "No complete training batch; reduce --batch-size or add more samples."
        )

    reparameterize = bool(
        profile_value(args.reparameterize, profile, "reparameterize", qat)
    )
    lsq = bool(config_value(args.lsq, profile.training.get("lsq", False) if profile else False))
    insert_identity_bn = bool(
        config_value(
            args.insert_identity_bn,
            profile.training.get("insert_identity_bn", False) if profile else False,
        )
    )
    freeze_bn_stats = False
    if lsq and not qat:
        raise ValueError("--lsq is only valid for QAT training.")
    if insert_identity_bn and not qat:
        raise ValueError("--insert-identity-bn is only valid for QAT training.")
    if lsq:
        if not qat_config or not is_lsq_config(qat_config):
            raise ValueError(
                "--lsq requires a QAT JSON with top-level 'lsq': true; "
                "got qat_config={!r}".format(qat_config)
            )
    kd = bool(config_value(args.kd, profile.training.get("kd", False) if profile else False))
    teacher_weights = config_value(
        args.teacher_weights,
        profile.training.get("teacher_weights") if profile else None,
    )
    teacher_model_config = config_value(
        args.teacher_model_config,
        profile.training.get("teacher_model_config") if profile else None,
    )
    kd_mode = profile_value(args.kd_mode, profile, "kd_mode")
    kd_weight = profile_value(args.kd_weight, profile, "kd_weight", 1.0)
    kd_temperature = profile_value(
        args.kd_temperature,
        profile,
        "kd_temperature",
        4.0,
    )
    kd_neck_weight = profile_value(
        args.kd_neck_weight,
        profile,
        "kd_neck_weight",
        0.0,
    )
    kd_backbone_weight = profile_value(
        args.kd_backbone_weight,
        profile,
        "kd_backbone_weight",
        0.0,
    )
    if kd:
        if not args.weights:
            raise ValueError("KD training requires --weights for the teacher baseline.")
        if kd_weight < 0 or kd_neck_weight < 0 or kd_backbone_weight < 0:
            raise ValueError("KD weights must be non-negative.")
        if kd_temperature <= 0:
            raise ValueError("--kd-temperature must be positive.")
    requested_amp = bool(profile_value(args.amp, profile, "amp", True))
    effective_amp = bool(requested_amp and not qat)
    if qat and requested_amp:
        warnings.warn(
            "AMP is disabled for PT2E QAT because fused fake-quant observers "
            "require FP32 activations.",
            stacklevel=2,
        )
    model = build_task_model(
        args.task,
        args.model_config,
        weights_path=args.weights,
        reparameterize=reparameterize,
        det_graph="training" if qat else "pretrained_train",
        rec_ctc_backbone_grad=rec_ctc_backbone_grad,
        rec_graph=rec_graph,
        insert_identity_bn=insert_identity_bn,
    )
    if kd:
        _set_intermediate_exposure(model, args.task, qat, rec_graph)
    float_node_count = None
    if qat:
        example_images, example_targets = next(iter(loader))
        example_inputs = (example_images,)
        batch_aligned_inputs = 0
        if rec_multi_head:
            model = FullRecTrainingWrapper(
                model,
                max_text_length=int(global_config.get("max_text_length", 25)),
                expose_intermediates=kd,
            ).set_qat_capture_mode()
            example_inputs = (example_images, example_targets["gtc_targets"])
            batch_aligned_inputs = 1
        dynamic_batch_max = (
            max(batch for _, batch in batch_sampler.scale_batches)
            if multi_scale_training
            else batch_size
        )
        if insert_identity_bn:
            bn_statistics_steps = int(
                profile_value(
                    args.bn_statistics_steps,
                    profile,
                    "bn_statistics_steps",
                    200,
                )
            )
            if bn_statistics_steps <= 0:
                raise ValueError("--bn-statistics-steps must be positive.")
            bn_training_momentum = profile_value(
                args.bn_training_momentum,
                profile,
                "bn_training_momentum",
                None,
            )
            if bn_training_momentum is not None:
                bn_training_momentum = float(bn_training_momentum)
                if not 0.0 < bn_training_momentum <= 1.0:
                    raise ValueError(
                        "--bn-training-momentum must be in (0, 1]."
                    )
            freeze_bn_stats = bool(
                config_value(
                    args.freeze_bn_stats,
                    profile.training.get("freeze_bn_stats", False)
                    if profile
                    else False,
                )
            )
        if insert_identity_bn and not args.eval_only and not args.resume:
            model = model.to(args.device)
            initialize_kept_bn_statistics(
                model,
                loader,
                bn_statistics_steps,
                rec_multi_head=rec_multi_head,
                device=args.device,
                training_momentum=bn_training_momentum,
                freeze_bn_stats=freeze_bn_stats,
            )
            model = model.cpu()
        quantizer = load_axera_quantizer(qat_config)
        model, float_node_count = prepare_qat_model(
            model,
            example_inputs,
            quantizer,
            freeze_kept_bn_stats=freeze_bn_stats,
            dynamic_shapes=build_qat_dynamic_shapes(
                example_images,
                dynamic_batch=example_images.shape[0] > 1,
                dynamic_heights=dynamic_heights,
                batch_aligned_inputs=batch_aligned_inputs,
                max_batch=dynamic_batch_max,
            ),
        )
        if rec_multi_head:
            model.graph_role = "pretrained_train"
            model.model_type = "rec"
            model.output_names = FullRecTrainingWrapper.output_names

        if lsq and not args.eval_only:
            _prepare_lsq_training(
                model,
                loader,
                rec_multi_head,
                resume=bool(args.resume),
            )

    optimizer_override = profile_value(args.optimizer, profile, "optimizer")
    momentum_override = profile_value(args.momentum, profile, "momentum")
    if qat and optimizer_override not in (None, "AdamW", "SGD", "Momentum"):
        raise ValueError(
            "QAT optimizer must be AdamW or SGD (momentum); Adam's L2 weight "
            f"decay corrupts LSQ learnable scales. Got: {optimizer_override}"
        )
    optimizer = build_optimizer(
        model,
        config,
        learning_rate=profile_value(
            args.learning_rate, profile, "learning_rate"
        ),
        weight_decay=profile_value(args.weight_decay, profile, "weight_decay"),
        lab_lr_multiplier=profile_value(
            args.lab_lr_multiplier, profile, "lab_lr_multiplier"
        ),
        ctc_fc_weight_decay=profile_value(
            args.ctc_fc_weight_decay,
            profile,
            "ctc_fc_weight_decay",
        ),
        optimizer_name=optimizer_override,
        momentum=momentum_override,
    )
    warmup_epochs = profile_value(
        args.warmup_epochs, profile, "warmup_epochs"
    )
    lr_final_factor = profile_value(
        args.lr_final_factor, profile, "lr_final_factor", 0.0
    )
    scheduler = build_scheduler(
        optimizer,
        config,
        epochs,
        warmup_epochs=warmup_epochs,
        final_factor=lr_final_factor,
    )
    teacher = None
    kd_criterion = None
    if kd:
        teacher = build_teacher(
            args,
            qat,
            reparameterize,
            rec_graph,
            rec_ctc_backbone_grad,
        )
        kd_criterion = build_kd_criterion(
            args.task,
            kd_layers=build_kd_layers(args, kd_mode),
            kd_weight=kd_weight,
            kd_temperature=kd_temperature,
            rec_graph=rec_graph,
        )
    trainer = Trainer(
        model,
        build_criterion(args.task, config, rec_multi_head=rec_multi_head),
        optimizer,
        scheduler=scheduler,
        device=args.device,
        amp=effective_amp,
        grad_clip_norm=profile_value(
            args.grad_clip_norm, profile, "grad_clip_norm"
        ),
        teacher=teacher,
        kd_criterion=kd_criterion,
        kd_weight=kd_weight if kd else 1.0,
        freeze_kept_bn_stats=freeze_bn_stats,
    )
    validation_metric = (
        build_validation_metric(args.task, config, config_path=args.model_config)
        if validation_loader is not None
        else None
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_configs = copy_run_configs(
        output_dir,
        model_config=args.model_config,
        training_profile=profile.path if profile is not None else None,
        qat_config=qat_config if qat else None,
        teacher_model_config=(
            teacher_model_config if kd and teacher_model_config else None
        ),
    )
    run_metadata = {
        "task": args.task,
        "model_config": str(Path(args.model_config).resolve()),
        "weights": str(Path(args.weights).resolve()) if args.weights else None,
        "label_file": str(Path(args.label_file).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "val_label_file": (
            str(Path(args.val_label_file).resolve())
            if args.val_label_file
            else None
        ),
        "val_data_dir": (
            str(Path(args.val_data_dir or args.data_dir).resolve())
            if args.val_label_file
            else None
        ),
        "training_profile": str(profile.path) if profile is not None else None,
        "training_profile_name": profile.name if profile is not None else None,
        "training_profile_sha256": profile.sha256 if profile is not None else None,
        "qat": qat,
        "qat_config": str(Path(qat_config).resolve()) if qat_config else None,
        "qat_config_sha256": file_sha256(qat_config) if qat_config else None,
        "copied_configs": {
            str(Path(source).name): str(destination)
            for source, destination in copied_configs
        },
        "torch_version": str(torch.__version__),
        "reparameterized": reparameterize,
        "lsq": lsq,
        "insert_identity_bn": insert_identity_bn,
        "bn_statistics_steps": bn_statistics_steps if insert_identity_bn else None,
        "bn_training_momentum": (
            bn_training_momentum if insert_identity_bn else None
        ),
        "freeze_bn_stats": freeze_bn_stats if insert_identity_bn else None,
        "kd": kd,
        "kd_mode": kd_mode if kd else None,
        "kd_weight": kd_weight if kd else None,
        "kd_temperature": kd_temperature if kd else None,
        "kd_neck_weight": kd_neck_weight if kd else None,
        "kd_backbone_weight": kd_backbone_weight if kd else None,
        "teacher_weights": (
            str(Path(teacher_weights).resolve())
            if kd and teacher_weights
            else None
        ),
        "teacher_model_config": (
            str(Path(teacher_model_config).resolve())
            if kd and teacher_model_config
            else None
        ),
        "rec_ctc_backbone_grad": rec_ctc_backbone_grad,
        "rec_graph": rec_graph if args.task == "rec" else None,
        "head_schema": (
            ["ctc", "ctc_neck", "gtc"]
            if rec_multi_head
            else ["ctc"]
            if args.task == "rec"
            else ["maps"]
        ),
        "loss_schema": (
            ["CTCLoss", "NRTRLoss"]
            if rec_multi_head
            else ["CTCLoss"]
            if args.task == "rec"
            else ["DBLoss"]
        ),
        "seed": seed,
        "image_shape": list(image_shape),
        "batch_size": batch_size,
        "epochs": epochs,
        "warmup_epochs": int(
            config_value(
                warmup_epochs,
                config.get("Optimizer", {}).get("lr", {}).get("warmup_epoch", 0),
            )
        ),
        "lr_final_factor": float(lr_final_factor),
        "initial_learning_rate": float(optimizer.defaults["lr"]),
        "weight_decay": float(optimizer.defaults.get("weight_decay", 0.0)),
        "optimizer": optimizer.__class__.__name__.lower(),
        "optimizer_momentum": float(optimizer.defaults.get("momentum", 0.0)),
        "optimizer_param_groups": [
            {
                "name": group.get("group_name", "default"),
                "parameters": len(group["params"]),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
            }
            for group in optimizer.param_groups
        ],
        "observer_freeze_epoch": observer_freeze_epoch,
        "qat_ema": False,
        "preprocessing": (
            "deterministic_center_letterbox" if args.task == "det" else "rec_padding"
        ),
        "amp": effective_amp,
        "dynamic_batch": qat and batch_size > 1,
        "dynamic_batch_max": dynamic_batch_max if qat else None,
        "dynamic_heights": dynamic_heights or [],
        "augmentation": augmentation,
        "multi_scale_training": multi_scale_training,
        "float_nodes": float_node_count,
        "validation_main_indicator": (
            validation_metric.main_indicator
            if validation_metric is not None
            else None
        ),
        "float_accuracy_baseline": float_accuracy_baseline,
        "epoch2_max_accuracy_drop": epoch2_max_accuracy_drop,
    }
    if args.resume:
        saved_metadata = trainer.load_checkpoint(args.resume)
        validate_resume_contract(saved_metadata, run_metadata)
    if args.eval_only:
        if args.eval_stage == "converted":
            trainer.model = convert_prepared_model(trainer.model).to(trainer.device)
        validation = trainer.evaluate(
            validation_loader,
            metric=validation_metric,
        )
        summary = {
            "checkpoint": (
                str(Path(args.resume).resolve()) if args.resume else None
            ),
            "weights": str(Path(args.weights).resolve()) if args.weights else None,
            "eval_stage": args.eval_stage,
            "epoch": trainer.epoch,
            "global_step": trainer.global_step,
            **{f"val_{name}": value for name, value in validation.items()},
        }
        accuracy_guard = epoch2_accuracy_guard(
            trainer.epoch,
            validation,
            float_accuracy_baseline=float_accuracy_baseline,
            max_accuracy_drop=epoch2_max_accuracy_drop,
        )
        if accuracy_guard is not None:
            summary["epoch2_accuracy_guard"] = accuracy_guard
        print(
            json.dumps(_round_summary_floats(summary), sort_keys=True),
            flush=True,
        )
        return summary
    start_epoch = trainer.epoch
    for epoch in range(start_epoch, epochs):
        current_learning_rates = {
            group.get("group_name", f"group_{index}"): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        current_learning_rate = current_learning_rates.get(
            "default", float(optimizer.param_groups[0]["lr"])
        )
        if (
            qat
            and observer_freeze_epoch is not None
            and epoch >= observer_freeze_epoch
            and trainer.global_step > 0
            and not trainer.observers_frozen
        ):
            trainer.freeze_observers()
        totals = {}
        for images, targets in loader:
            losses = trainer.train_step(images, targets)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value
            if (
                qat
                and observer_freeze_epoch is not None
                and epoch >= observer_freeze_epoch
                and not trainer.observers_frozen
            ):
                # A fresh PT2E QAT graph needs one batch to initialize
                # per-channel qparams before observers can be disabled.
                trainer.freeze_observers()
        trainer.epoch = epoch + 1
        trainer.step_scheduler()
        validation = (
            trainer.evaluate(validation_loader, metric=validation_metric)
            if validation_loader is not None
            else {}
        )
        summary = {
            "epoch": trainer.epoch,
            "steps": len(loader),
            "learning_rate": current_learning_rate,
            "learning_rates": current_learning_rates,
            "observers_frozen": trainer.observers_frozen,
            **{name: value / len(loader) for name, value in totals.items()},
            **{f"val_{name}": value for name, value in validation.items()},
        }
        accuracy_guard = epoch2_accuracy_guard(
            trainer.epoch,
            validation,
            float_accuracy_baseline=float_accuracy_baseline,
            max_accuracy_drop=epoch2_max_accuracy_drop,
        )
        if accuracy_guard is not None:
            summary["epoch2_accuracy_guard"] = accuracy_guard
        print(
            json.dumps(_round_summary_floats(summary), sort_keys=True),
            flush=True,
        )
        is_best = update_best_validation(
            trainer,
            validation,
            metric=validation_metric,
        )
        if trainer.epoch % save_every == 0:
            trainer.save_checkpoint(
                output_dir / f"epoch_{trainer.epoch:04d}.pt",
                metadata=run_metadata,
            )
        trainer.save_checkpoint(output_dir / "last.pt", metadata=run_metadata)
        if is_best:
            trainer.save_checkpoint(output_dir / "best.pt", metadata=run_metadata)
        if accuracy_guard is not None and accuracy_guard["triggered"]:
            guard_path = output_dir / "epoch2_accuracy_guard.json"
            guard_path.write_text(
                json.dumps(accuracy_guard, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            trainer.save_checkpoint(
                output_dir / "debug_epoch_0002.pt",
                metadata=run_metadata,
            )
            raise RuntimeError(
                "Epoch-2 recognition accuracy guard triggered: "
                f"float={accuracy_guard['float_accuracy_baseline']:.6f}, "
                f"qat={accuracy_guard['validation_accuracy']:.6f}, "
                f"drop={accuracy_guard['accuracy_drop']:.6f}, "
                f"limit={accuracy_guard['max_accuracy_drop']:.6f}."
            )


if __name__ == "__main__":
    # Debug aid: SIGUSR1 dumps all Python thread stacks to stderr. Useful to
    # locate deadlocks in the training loop (DataLoader / CUDA sync).
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    train(parse_args())
