from pathlib import Path

import torch

from pytorchocr.quantization import (
    FullRecTrainingWrapper,
    build_qat_dynamic_shapes,
    load_axera_quantizer,
    prepare_qat_model,
)
from pytorchocr.training import (
    build_task_model,
    load_ocr_config,
    relocate_checkpoint_metadata,
)


def load_qat_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = relocate_checkpoint_metadata(checkpoint.get("metadata", {}))
    if not metadata.get("qat", False):
        raise ValueError("Checkpoint is not marked as an Axera QAT checkpoint.")
    return checkpoint, metadata


def build_prepared_qat_checkpoint(
    metadata,
    *,
    weights_path=None,
    model_config=None,
    qat_config=None,
):
    metadata = relocate_checkpoint_metadata(metadata)
    model_config = str(Path(model_config or metadata["model_config"]))
    qat_config = str(Path(qat_config or metadata["qat_config"]))
    if weights_path is None:
        weights_path = metadata.get("weights")
    task = metadata["task"]
    rec_graph = metadata.get("rec_graph", "deploy")
    kd = bool(metadata.get("kd", False))
    model = build_task_model(
        task,
        model_config,
        weights_path=weights_path,
        reparameterize=bool(metadata.get("reparameterized", True)),
        det_graph="training",
        rec_ctc_backbone_grad=bool(
            metadata.get("rec_ctc_backbone_grad", False)
        ),
        rec_graph=rec_graph,
        insert_identity_bn=bool(metadata.get("insert_identity_bn", False)),
    )
    if kd and task == "det":
        # KD checkpoints were captured with intermediate features exposed.
        model.expose_intermediates = True
    captured_batch = int(metadata.get("batch_size", 1))
    image_shape = tuple(metadata["image_shape"])
    capture_images = torch.empty(captured_batch, *image_shape)
    dynamic_batch = bool(metadata.get("dynamic_batch", False))
    dynamic_heights = metadata.get("dynamic_heights") or None
    example_inputs = (capture_images,)
    batch_aligned_inputs = 0
    if task == "rec" and rec_graph == "pretrained_train":
        config = load_ocr_config(model_config)
        max_text_length = int(config["Global"].get("max_text_length", 25))
        model = FullRecTrainingWrapper(
            model,
            max_text_length=max_text_length,
            expose_intermediates=kd,
        ).set_qat_capture_mode()
        capture_targets = torch.zeros(
            captured_batch,
            max_text_length,
            dtype=torch.long,
        )
        example_inputs = (capture_images, capture_targets)
        batch_aligned_inputs = 1
    prepared, float_node_count = prepare_qat_model(
        model,
        example_inputs,
        load_axera_quantizer(qat_config),
        freeze_kept_bn_stats=bool(metadata.get("freeze_bn_stats", False)),
        dynamic_shapes=build_qat_dynamic_shapes(
            capture_images,
            dynamic_batch=dynamic_batch,
            dynamic_heights=dynamic_heights,
            batch_aligned_inputs=batch_aligned_inputs,
            max_batch=metadata.get("dynamic_batch_max"),
        ),
    )
    if task == "rec" and rec_graph == "pretrained_train":
        prepared.graph_role = "pretrained_train"
        prepared.model_type = "rec"
        prepared.output_names = FullRecTrainingWrapper.output_names
    return prepared, float_node_count
