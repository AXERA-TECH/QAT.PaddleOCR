from pytorchocr.postprocess.db_postprocess import DBPostProcess

from ..model_builder import resolve_config_path
from .detection import DetectionIoUEvaluator, DetectionMetric
from .recognition import CTCRecognitionMetric, levenshtein_distance


def build_validation_metric(task, config, config_path=None):
    metric_config = dict(config.get("Metric", {}))
    metric_name = metric_config.pop("name", None)
    main_indicator = metric_config.pop(
        "main_indicator",
        "hmean" if task == "det" else "acc",
    )
    if task == "det":
        if metric_name not in (None, "DetMetric"):
            raise ValueError(f"Unsupported detection metric: {metric_name}")
        post_config = dict(config.get("PostProcess", {}))
        post_name = post_config.pop("name", "DBPostProcess")
        if post_name != "DBPostProcess":
            raise ValueError(f"Unsupported detection postprocess: {post_name}")
        return DetectionMetric(
            DBPostProcess(**post_config),
            main_indicator=main_indicator,
        )
    if task == "rec":
        if metric_name not in (None, "RecMetric"):
            raise ValueError(f"Unsupported recognition metric: {metric_name}")
        global_config = config["Global"]
        dictionary_path = resolve_config_path(
            global_config["character_dict_path"],
            config_path=config_path,
        )
        return CTCRecognitionMetric(
            dictionary_path,
            use_space_char=global_config.get("use_space_char", False),
            main_indicator=main_indicator,
            ignore_space=metric_config.pop("ignore_space", True),
            is_filter=metric_config.pop("is_filter", False),
        )
    raise ValueError(f"Unsupported OCR task: {task}")


__all__ = [
    "CTCRecognitionMetric",
    "DetectionIoUEvaluator",
    "DetectionMetric",
    "build_validation_metric",
    "levenshtein_distance",
]
