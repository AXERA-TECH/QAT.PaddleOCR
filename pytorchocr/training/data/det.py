# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import json
from pathlib import Path

import cv2
import numpy as np
import pyclipper
import torch
from shapely.geometry import Polygon
from torch.utils.data import Dataset

from .augmentation import (
    DetectionSample,
    PaddleDetectionAugmentation,
)


class DBMapGenerator:
    def __init__(
        self,
        shrink_ratio=0.4,
        min_text_size=8,
        threshold_min=0.3,
        threshold_max=0.7,
    ):
        self.shrink_ratio = shrink_ratio
        self.min_text_size = min_text_size
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

    def __call__(self, image_shape, polygons, ignore_tags):
        height, width = image_shape
        polygons, ignore_tags = self._validate_polygons(
            polygons,
            ignore_tags.copy(),
            height,
            width,
        )
        threshold_map = np.zeros((height, width), dtype=np.float32)
        threshold_mask = np.zeros((height, width), dtype=np.float32)
        for polygon, ignored in zip(polygons, ignore_tags):
            if not ignored:
                self._draw_border_map(polygon, threshold_map, threshold_mask)
        threshold_map = (
            threshold_map * (self.threshold_max - self.threshold_min)
            + self.threshold_min
        )

        shrink_map = np.zeros((height, width), dtype=np.float32)
        shrink_mask = np.ones((height, width), dtype=np.float32)
        for index, polygon in enumerate(polygons):
            polygon_height = np.ptp(polygon[:, 1])
            polygon_width = np.ptp(polygon[:, 0])
            if (
                ignore_tags[index]
                or min(polygon_height, polygon_width) < self.min_text_size
            ):
                cv2.fillPoly(shrink_mask, [polygon.astype(np.int32)], 0)
                continue
            polygon_shape = Polygon(polygon)
            if polygon_shape.length <= 0 or not np.isfinite(polygon).all():
                cv2.fillPoly(shrink_mask, [polygon.astype(np.int32)], 0)
                continue
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(
                [tuple(point) for point in polygon],
                pyclipper.JT_ROUND,
                pyclipper.ET_CLOSEDPOLYGON,
            )
            shrunk = []
            for ratio in np.arange(self.shrink_ratio, 1, self.shrink_ratio):
                distance = (
                    polygon_shape.area
                    * (1 - ratio**2)
                    / polygon_shape.length
                )
                shrunk = padding.Execute(-distance)
                if len(shrunk) == 1:
                    break
            if not shrunk:
                cv2.fillPoly(shrink_mask, [polygon.astype(np.int32)], 0)
                continue
            for shrunk_polygon in shrunk:
                points = np.asarray(shrunk_polygon, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(shrink_map, [points], 1)

        return {
            "threshold_map": threshold_map,
            "threshold_mask": threshold_mask,
            "shrink_map": shrink_map,
            "shrink_mask": shrink_mask,
        }

    @staticmethod
    def _polygon_area(polygon):
        area = 0.0
        previous = polygon[-1]
        for point in polygon:
            area += point[0] * previous[1] - point[1] * previous[0]
            previous = point
        return area / 2.0

    def _validate_polygons(self, polygons, ignore_tags, height, width):
        for index, polygon in enumerate(polygons):
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            area = self._polygon_area(polygon)
            if abs(area) < 1 or not np.isfinite(polygon).all():
                ignore_tags[index] = True
            if area > 0:
                polygons[index] = polygon[::-1]
        return polygons, ignore_tags

    def _draw_border_map(self, polygon, canvas, mask):
        polygon_shape = Polygon(polygon)
        if polygon_shape.area <= 0 or polygon_shape.length <= 0:
            return
        distance = (
            polygon_shape.area
            * (1 - self.shrink_ratio**2)
            / polygon_shape.length
        )
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(
            [tuple(point) for point in polygon],
            pyclipper.JT_ROUND,
            pyclipper.ET_CLOSEDPOLYGON,
        )
        padded = padding.Execute(distance)
        if not padded:
            return
        padded = np.asarray(padded[0], dtype=np.int32)
        cv2.fillPoly(mask, [padded], 1.0)

        xmin, ymin = padded.min(axis=0)
        xmax, ymax = padded.max(axis=0)
        width = int(xmax - xmin + 1)
        height = int(ymax - ymin + 1)
        local_polygon = polygon.copy()
        local_polygon[:, 0] -= xmin
        local_polygon[:, 1] -= ymin
        xs = np.broadcast_to(np.arange(width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.arange(height).reshape(height, 1), (height, width))
        distances = np.zeros(
            (local_polygon.shape[0], height, width),
            dtype=np.float32,
        )
        for index in range(local_polygon.shape[0]):
            next_index = (index + 1) % local_polygon.shape[0]
            distances[index] = np.clip(
                self._distance(
                    xs,
                    ys,
                    local_polygon[index],
                    local_polygon[next_index],
                )
                / max(distance, 1.0e-6),
                0,
                1,
            )
        distance_map = distances.min(axis=0)
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        source = distance_map[
            ymin_valid - ymin : ymax_valid - ymax + height,
            xmin_valid - xmin : xmax_valid - xmax + width,
        ]
        target = canvas[ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1]
        canvas[ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1] = (
            np.fmax(1 - source, target)
        )

    @staticmethod
    def _distance(xs, ys, point_1, point_2):
        square_1 = (xs - point_1[0]) ** 2 + (ys - point_1[1]) ** 2
        square_2 = (xs - point_2[0]) ** 2 + (ys - point_2[1]) ** 2
        segment_square = (point_1[0] - point_2[0]) ** 2 + (
            point_1[1] - point_2[1]
        ) ** 2
        if segment_square <= 0:
            return np.sqrt(square_1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine = (segment_square - square_1 - square_2) / (
                2 * np.sqrt(square_1 * square_2)
            )
            sine_square = np.nan_to_num(1 - cosine**2)
            result = np.sqrt(square_1 * square_2 * sine_square / segment_square)
        endpoints = np.sqrt(np.minimum(square_1, square_2))
        result[cosine < 0] = endpoints[cosine < 0]
        return result


class DetectionDataset(Dataset):
    def __init__(
        self,
        label_file,
        data_dir=".",
        image_shape=(3, 640, 640),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        padding_value=114,
        map_generator=None,
        return_polygons=False,
        augmentation="none",
        det_preprocess="paddle",
    ):
        self.label_file = Path(label_file).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.image_shape = tuple(image_shape)
        if len(self.image_shape) != 3 or self.image_shape[0] != 3:
            raise ValueError("Detection images must have three channels.")
        if self.image_shape[1] <= 0 or self.image_shape[2] <= 0:
            raise ValueError("Detection image height and width must be positive.")
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.padding_value = int(padding_value)
        if not 0 <= self.padding_value <= 255:
            raise ValueError("Detection padding value must be in [0, 255].")
        self.map_generator = map_generator or DBMapGenerator()
        self.return_polygons = bool(return_polygons)
        if det_preprocess not in ("letterbox", "paddle"):
            raise ValueError(
                "Detection preprocessing must be 'letterbox' or 'paddle'."
            )
        self.det_preprocess = det_preprocess
        if augmentation not in ("none", "paddle"):
            raise ValueError("Detection augmentation must be 'none' or 'paddle'.")
        self.augmentation = augmentation
        self.augmenter = (
            PaddleDetectionAugmentation(self.image_shape)
            if augmentation == "paddle"
            else None
        )
        with open(self.label_file, encoding="utf-8") as label_stream:
            self.samples = [line.rstrip("\r\n") for line in label_stream if line.strip()]
        if not self.samples:
            raise ValueError(f"Detection label file is empty: {self.label_file}")

    def _letterbox(self, image, polygons):
        _, target_height, target_width = self.image_shape
        source_height, source_width = image.shape[:2]
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = min(target_width, max(1, round(source_width * scale)))
        resized_height = min(target_height, max(1, round(source_height * scale)))
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        left = (target_width - resized_width) // 2
        right = target_width - resized_width - left
        top = (target_height - resized_height) // 2
        bottom = target_height - resized_height - top
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(self.padding_value,) * 3,
        )
        x_scale = resized_width / source_width
        y_scale = resized_height / source_height
        for polygon in polygons:
            polygon[:, 0] = polygon[:, 0] * x_scale + left
            polygon[:, 1] = polygon[:, 1] * y_scale + top
        return image

    def _paddle_resize(self, image, polygons):
        """Resize to the configured shape using PaddleOCR's fixed-shape path."""
        _, target_height, target_width = self.image_shape
        source_height, source_width = image.shape[:2]
        image = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        x_scale = target_width / source_width
        y_scale = target_height / source_height
        for polygon in polygons:
            polygon[:, 0] *= x_scale
            polygon[:, 1] *= y_scale
        return image

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, index):
        line = self.samples[index]
        try:
            image_name, encoded = line.split("\t", 1)
            annotations = json.loads(encoded)
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid detection label at line {index + 1}") from error
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = self.data_dir / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not decode detection image: {image_path}")
        polygons = []
        ignore_tags = []
        texts = []
        for annotation in annotations:
            points = np.asarray(annotation["points"], dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                raise ValueError(f"Invalid polygon in {image_path}")
            polygons.append(points)
            text = str(annotation.get("transcription", ""))
            ignore_tags.append(text in ("*", "###"))
            texts.append(text)
        return DetectionSample(image, polygons, ignore_tags, texts)

    def __getitem__(self, index):
        sample = self._load_sample(index)
        if self.augmenter is not None:
            external = None
            if len(self.samples) > 1:
                external_index = np.random.randint(len(self.samples) - 1)
                if external_index >= index:
                    external_index += 1
                external = self._load_sample(external_index)
            sample = self.augmenter(sample, external)
        image = sample.image
        polygons = sample.polygons
        ignore_tags = sample.ignore_tags

        _, target_height, target_width = self.image_shape
        if self.det_preprocess == "letterbox":
            image = self._letterbox(image, polygons)
        else:
            image = self._paddle_resize(image, polygons)
        maps = self.map_generator(
            (target_height, target_width),
            polygons,
            np.asarray(ignore_tags, dtype=np.bool_),
        )
        image = image.astype(np.float32) / 255.0
        image = ((image - self.mean) / self.std).transpose(2, 0, 1)
        targets = {
            name: torch.from_numpy(value)
            for name, value in maps.items()
        }
        if self.return_polygons:
            targets.update(
                {
                    "polygons": [polygon.copy() for polygon in polygons],
                    "ignore_tags": np.asarray(ignore_tags, dtype=np.bool_),
                    "shape": np.asarray(
                        [target_height, target_width, 1.0, 1.0],
                        dtype=np.float32,
                    ),
                }
            )
        return torch.from_numpy(np.ascontiguousarray(image)), targets


def detection_collate(batch):
    """Stack fixed DB maps while preserving each image's variable polygon list."""
    images, targets = zip(*batch)
    expected_keys = set(targets[0])
    if any(set(item) != expected_keys for item in targets[1:]):
        raise ValueError("Detection batch targets have inconsistent keys.")
    collated_targets = {}
    for key in targets[0]:
        values = [item[key] for item in targets]
        collated_targets[key] = (
            torch.stack(values) if torch.is_tensor(values[0]) else values
        )
    return torch.stack(images), collated_targets
