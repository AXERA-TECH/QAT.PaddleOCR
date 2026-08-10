import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from pytorchocr.training.model_builder import load_ocr_config, resolve_config_path
from pytorchocr.utils.hashing import file_sha256


def json_hash(value):
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_image(data_dir, image_name):
    image_path = Path(image_name)
    if not image_path.is_absolute():
        image_path = Path(data_dir) / image_path
    return image_path.resolve()


def image_info(path):
    path = Path(path)
    if not path.is_file():
        return {"exists": False, "path": str(path)}
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "exists": True,
            "decodable": False,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
    height, width = image.shape[:2]
    return {
        "exists": True,
        "decodable": True,
        "height": int(height),
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "width": int(width),
    }


def distribution(values):
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "p99": float(np.quantile(array, 0.99)),
    }


def stable_debug_selection(records, count, score_keys):
    if count <= 0 or not records:
        return []
    selected = []

    def add(record):
        selected_lines = {item["line"] for item in selected}
        if record is not None and record["line"] not in selected_lines:
            selected.append(record)

    for key in score_keys:
        candidates = [record for record in records if record.get(key) is not None]
        if candidates:
            add(min(candidates, key=lambda item: (item[key], item["line"])))
            add(max(candidates, key=lambda item: (item[key], -item["line"])))
    ordered = sorted(records, key=lambda item: item["line"])
    remaining = max(0, count - len(selected))
    if remaining:
        indices = np.linspace(0, len(ordered) - 1, remaining, dtype=int)
        for index in indices:
            add(ordered[int(index)])
    for record in ordered:
        if len(selected) >= count:
            break
        add(record)
    return sorted(selected[:count], key=lambda item: item["line"])


def audit_detection_label(label_path, data_dir, debug_count):
    records = []
    errors = []
    polygon_count = 0
    care_count = 0
    ignore_count = 0
    empty_images = 0
    aspect_ratios = []
    image_areas = []
    polygon_short_sides = []
    manifest = []
    with open(label_path, encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            try:
                image_name, encoded = line.split("\t", 1)
                annotations = json.loads(encoded)
                if not isinstance(annotations, list):
                    raise ValueError("annotations must be a list")
            except (ValueError, json.JSONDecodeError) as error:
                errors.append({"line": line_number, "error": str(error)})
                continue
            path = resolve_image(data_dir, image_name)
            info = image_info(path)
            manifest.append(info)
            valid_polygons = 0
            ignored = 0
            for annotation in annotations:
                points = np.asarray(annotation.get("points", []), dtype=np.float32)
                if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                    errors.append({"line": line_number, "error": "invalid polygon"})
                    continue
                valid_polygons += 1
                polygon_count += 1
                is_ignored = annotation.get("transcription") in ("*", "###")
                ignored += int(is_ignored)
                ignore_count += int(is_ignored)
                care_count += int(not is_ignored)
                polygon_short_sides.append(
                    float(min(np.ptp(points[:, 0]), np.ptp(points[:, 1])))
                )
            empty_images += int(valid_polygons == 0)
            if info.get("decodable"):
                aspect = info["width"] / info["height"]
                area = info["width"] * info["height"]
                aspect_ratios.append(aspect)
                image_areas.append(area)
            else:
                aspect = None
                area = None
            records.append(
                {
                    "aspect_ratio": aspect,
                    "image": image_name,
                    "image_area": area,
                    "ignore_polygons": ignored,
                    "line": line_number,
                    "polygons": valid_polygons,
                }
            )
    debug = stable_debug_selection(
        records,
        debug_count,
        ("aspect_ratio", "image_area", "polygons", "ignore_polygons"),
    )
    return {
        "care_polygons": care_count,
        "debug_samples": debug,
        "empty_annotation_images": empty_images,
        "errors": errors,
        "ignore_polygons": ignore_count,
        "image_area": distribution(image_areas),
        "image_aspect_ratio": distribution(aspect_ratios),
        "image_manifest_sha256": json_hash(manifest),
        "label_file": str(Path(label_path).resolve()),
        "label_sha256": file_sha256(label_path),
        "missing_images": sum(not item.get("exists", False) for item in manifest),
        "polygon_count": polygon_count,
        "polygon_short_side": distribution(polygon_short_sides),
        "sample_count": len(records),
        "undecodable_images": sum(
            item.get("exists", False) and not item.get("decodable", False)
            for item in manifest
        ),
    }


def load_characters(dictionary_path, use_space_char):
    with open(dictionary_path, encoding="utf-8") as stream:
        characters = [line.rstrip("\r\n") for line in stream]
    if use_space_char:
        characters.append(" ")
    return characters


def audit_recognition_label(
    label_path,
    data_dir,
    characters,
    max_text_length,
    debug_count,
):
    known = set(characters)
    records = []
    errors = []
    manifest = []
    unknown_counter = Counter()
    character_counter = Counter()
    text_lengths = []
    empty_texts = 0
    overlength_texts = 0
    loader_accepted = 0
    strictly_encodable = 0
    with open(label_path, encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            try:
                image_name, text = line.split("\t", 1)
            except ValueError as error:
                errors.append({"line": line_number, "error": str(error)})
                continue
            path = resolve_image(data_dir, image_name)
            info = image_info(path)
            manifest.append(info)
            unknown = sorted({character for character in text if character not in known})
            unknown_counter.update(character for character in text if character not in known)
            character_counter.update(text)
            length = len(text)
            text_lengths.append(length)
            empty_texts += int(not text)
            overlength_texts += int(length > max_text_length)
            known_character_count = sum(character in known for character in text)
            is_loader_accepted = bool(text) and length <= max_text_length and known_character_count > 0
            is_strictly_encodable = is_loader_accepted and not unknown
            loader_accepted += int(is_loader_accepted)
            strictly_encodable += int(is_strictly_encodable)
            aspect = info["width"] / info["height"] if info.get("decodable") else None
            repeated = any(left == right for left, right in zip(text, text[1:]))
            records.append(
                {
                    "aspect_ratio": aspect,
                    "loader_accepted": is_loader_accepted,
                    "strictly_encodable": is_strictly_encodable,
                    "has_repeated_character": repeated,
                    "has_space": " " in text,
                    "image": image_name,
                    "line": line_number,
                    "text": text,
                    "text_length": length,
                    "unknown_characters": unknown,
                }
            )
    debug = stable_debug_selection(
        records,
        debug_count,
        ("aspect_ratio", "text_length", "has_space", "has_repeated_character"),
    )
    return {
        "character_coverage": len(character_counter),
        "debug_samples": debug,
        "empty_texts": empty_texts,
        "loader_accepted_samples": loader_accepted,
        "errors": errors,
        "image_manifest_sha256": json_hash(manifest),
        "label_file": str(Path(label_path).resolve()),
        "label_sha256": file_sha256(label_path),
        "missing_images": sum(not item.get("exists", False) for item in manifest),
        "overlength_texts": overlength_texts,
        "sample_count": len(records),
        "strictly_encodable_samples": strictly_encodable,
        "text_length": distribution(text_lengths),
        "undecodable_images": sum(
            item.get("exists", False) and not item.get("decodable", False)
            for item in manifest
        ),
        "unknown_character_count": sum(unknown_counter.values()),
        "unknown_characters": dict(sorted(unknown_counter.items())),
    }


def model_contract(task, config_path, paddle_weights, torch_weights):
    config = load_ocr_config(config_path)
    global_config = config.get("Global", {})
    architecture = config["Architecture"]
    contract = {
        "architecture": architecture,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        "image_shape": global_config.get("d2s_train_image_shape"),
        "model_name": global_config.get("model_name"),
        "paddle_weights": str(Path(paddle_weights).resolve()),
        "paddle_weights_sha256": file_sha256(paddle_weights),
        "task": task,
        "torch_weights": str(Path(torch_weights).resolve()),
        "torch_weights_sha256": file_sha256(torch_weights),
    }
    if task == "rec":
        dictionary_path = resolve_config_path(
            global_config["character_dict_path"],
            config_path=config_path,
        )
        characters = load_characters(
            dictionary_path,
            bool(global_config.get("use_space_char", False)),
        )
        contract.update(
            {
                "blank_index": 0,
                "character_count_without_blank": len(characters),
                "ctc_class_count": len(characters) + 1,
                "dictionary_path": str(dictionary_path),
                "dictionary_sha256": file_sha256(dictionary_path),
                "max_text_length": int(global_config.get("max_text_length", 25)),
                "use_space_char": bool(global_config.get("use_space_char", False)),
            }
        )
    return contract


def validate_audit(audit, name):
    failures = []
    for split in ("train", "val"):
        result = audit[split]
        if result["errors"]:
            failures.append(f"{name}.{split}: {len(result['errors'])} label errors")
        if result["missing_images"]:
            failures.append(f"{name}.{split}: {result['missing_images']} missing images")
        if result["undecodable_images"]:
            failures.append(
                f"{name}.{split}: {result['undecodable_images']} undecodable images"
            )
    return failures
