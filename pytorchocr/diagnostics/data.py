from pytorchocr.training import build_dataset, load_ocr_config


def build_diagnostic_dataset(
    task,
    metadata,
    label_file,
    data_dir,
    *,
    return_polygons=False,
):
    model_config = metadata["model_config"]
    return build_dataset(
        task,
        model_config,
        load_ocr_config(model_config),
        tuple(metadata["image_shape"]),
        label_file,
        data_dir,
        return_polygons=return_polygons,
        rec_multi_head=(
            task == "rec" and metadata.get("rec_graph") == "pretrained_train"
        ),
        # Checkpoints created before det_preprocess was recorded used the
        # historical centered-letterbox preprocessing.
        det_preprocess=metadata.get("det_preprocess", "letterbox"),
    )


def sample_ids(dataset, count):
    samples = getattr(dataset, "samples", None)
    if samples is None and hasattr(dataset, "dataset"):
        samples = getattr(dataset.dataset, "samples", None)
    if samples is None:
        return []
    identifiers = []
    for sample in samples[:count]:
        value = sample[0] if isinstance(sample, tuple) else sample.split("\t", 1)[0]
        identifiers.append(str(value))
    return identifiers
