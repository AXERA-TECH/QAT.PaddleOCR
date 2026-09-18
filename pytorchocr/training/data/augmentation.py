"""PaddleOCR-compatible training augmentations adapted from PytorchOCR."""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Polygon, box as shapely_box

from .text_image_aug import tia_distort, tia_perspective, tia_stretch


@dataclass(frozen=True)
class DetectionSample:
    image: np.ndarray
    polygons: list[np.ndarray]
    ignore_tags: list[bool]
    texts: list[str]


def _transform_polygons(polygons, matrix):
    transformed = []
    for polygon in polygons:
        homogeneous = np.concatenate(
            [polygon.astype(np.float32), np.ones((len(polygon), 1), np.float32)],
            axis=1,
        )
        transformed.append((homogeneous @ matrix.T).astype(np.float32))
    return transformed


def _fit_rotation(image, polygons, angle):
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    output_width = max(1, int(round(height * sine + width * cosine)))
    output_height = max(1, int(round(height * cosine + width * sine)))
    matrix[0, 2] += output_width / 2.0 - center[0]
    matrix[1, 2] += output_height / 2.0 - center[1]
    image = cv2.warpAffine(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return image, _transform_polygons(polygons, matrix)


def _is_outside(polygon, x, y, width, height):
    return bool(
        polygon[:, 0].max() < x
        or polygon[:, 0].min() > x + width
        or polygon[:, 1].max() < y
        or polygon[:, 1].min() > y + height
    )


def _is_inside(polygon, x, y, width, height):
    return bool(
        polygon[:, 0].min() >= x
        and polygon[:, 0].max() <= x + width
        and polygon[:, 1].min() >= y
        and polygon[:, 1].max() <= y + height
    )


def _minimum_rotated_side(polygon):
    if len(polygon) < 3:
        return 0.0
    return float(min(cv2.minAreaRect(polygon.astype(np.float32))[1]))


def _minimum_quad_side(polygon):
    if len(polygon) != 4:
        return 0.0
    return float(
        min(
            np.linalg.norm(polygon[index] - polygon[(index + 1) % 4])
            for index in range(4)
        )
    )


def _clip_polygon_to_rect(polygon, x, y, width, height):
    try:
        clipped = Polygon(polygon).intersection(
            shapely_box(x, y, x + width, y + height)
        )
        if clipped.is_empty:
            return None
        if clipped.geom_type == "Polygon":
            geometry = clipped
        elif clipped.geom_type in ("MultiPolygon", "GeometryCollection"):
            polygons = [
                item
                for item in clipped.geoms
                if item.geom_type == "Polygon" and not item.is_empty
            ]
            if not polygons:
                return None
            geometry = max(polygons, key=lambda item: item.area)
        else:
            return None
        coordinates = np.asarray(geometry.exterior.coords[:-1], dtype=np.float32)
        if len(coordinates) <= 3:
            return None
        if len(coordinates) == 4:
            return coordinates
        contour = coordinates.reshape(-1, 1, 2)
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1.0e-6:
            return None
        low, high = 0.0, 0.5
        best = None
        for _ in range(50):
            middle = (low + high) / 2
            approximation = cv2.approxPolyDP(contour, middle * perimeter, True)
            if len(approximation) <= 4:
                best = approximation
                high = middle
            else:
                low = middle
        if best is None or len(best) < 3:
            return None
        return best.reshape(-1, 2)
    except Exception:
        return None


def _rotate_patch(image, angle):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    output_width = max(1, int(round(height * sine + width * cosine)))
    output_height = max(1, int(round(height * cosine + width * sine)))
    matrix[0, 2] += output_width / 2.0 - width / 2.0
    matrix[1, 2] += output_height / 2.0 - height / 2.0
    rotated = cv2.warpAffine(image, matrix, (output_width, output_height))
    mask = cv2.warpAffine(
        np.full((height, width), 255, dtype=np.uint8),
        matrix,
        (output_width, output_height),
    )
    corners = np.asarray(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    return rotated, mask, _transform_polygons([corners], matrix)[0]


def _quadrilateral_crop(image, polygon):
    width = max(
        1,
        int(round(max(np.linalg.norm(polygon[0] - polygon[1]), np.linalg.norm(polygon[2] - polygon[3])))),
    )
    height = max(
        1,
        int(round(max(np.linalg.norm(polygon[0] - polygon[3]), np.linalg.norm(polygon[1] - polygon[2])))),
    )
    target = np.asarray(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(polygon.astype(np.float32), target)
    return cv2.warpPerspective(image, matrix, (width, height))


def _overlaps_existing(box, polygons):
    xmin, ymin = box.min(axis=0)
    xmax, ymax = box.max(axis=0)
    return any(
        not _is_outside(polygon, xmin, ymin, xmax - xmin, ymax - ymin)
        for polygon in polygons
    )


class PaddleDetectionAugmentation:
    """PP-OCR DB augmentation without the optional imgaug dependency."""

    def __init__(
        self,
        image_shape,
        *,
        additional_augmentations=True,
        copy_paste_ratio=0.2,
        flip_probability=0.5,
        rotate_probability=0.5,
        rotate_range=(-45.0, 45.0),
        scale_range=(0.1, 2.0),
        crop_tries=50,
        min_crop_side_ratio=0.1,
    ):
        self.target_height = int(image_shape[1])
        self.target_width = int(image_shape[2])
        self.additional_augmentations = bool(additional_augmentations)
        self.copy_paste_ratio = float(copy_paste_ratio)
        self.flip_probability = float(flip_probability)
        self.rotate_probability = float(rotate_probability)
        self.rotate_range = tuple(float(value) for value in rotate_range)
        self.scale_range = tuple(float(value) for value in scale_range)
        self.crop_tries = int(crop_tries)
        self.min_crop_side_ratio = float(min_crop_side_ratio)

    def _copy_paste(self, sample, external):
        if external is None:
            return sample
        candidates = [
            index
            for index, (polygon, ignored) in enumerate(
                zip(external.polygons, external.ignore_tags)
            )
            if not ignored and len(polygon) == 4
        ]
        random.shuffle(candidates)
        count = min(max(1, int(self.copy_paste_ratio * len(external.polygons))), 30)
        image = sample.image.copy()
        polygons = [polygon.copy() for polygon in sample.polygons]
        ignore_tags = list(sample.ignore_tags)
        texts = list(sample.texts)
        height, width = image.shape[:2]
        for index in candidates[:count]:
            patch = _quadrilateral_crop(external.image, external.polygons[index])
            patch, mask, box = _rotate_patch(patch, random.uniform(0.0, 360.0))
            patch_height, patch_width = patch.shape[:2]
            if patch_height > height or patch_width > width:
                continue
            location = None
            for _ in range(50):
                left = random.randint(0, width - patch_width)
                top = random.randint(0, height - patch_height)
                shifted = box + np.asarray([left, top], dtype=np.float32)
                if not _overlaps_existing(shifted, polygons):
                    location = (left, top, shifted)
                    break
            if location is None:
                continue
            left, top, shifted = location
            target = image[top : top + patch_height, left : left + patch_width]
            target[mask > 0] = patch[mask > 0]
            polygons.append(shifted)
            ignore_tags.append(bool(external.ignore_tags[index]))
            texts.append(external.texts[index])
        return DetectionSample(image, polygons, ignore_tags, texts)

    def _random_crop(self, sample):
        image = sample.image
        polygons = sample.polygons
        ignore_tags = sample.ignore_tags
        texts = sample.texts
        height, width = image.shape[:2]
        care_indices = [
            index for index, ignored in enumerate(ignore_tags) if not ignored
        ]

        if not care_indices:
            crop_x, crop_y, crop_width, crop_height = 0, 0, width, height
            valid_care = {}
        else:
            character_heights = {
                index: _minimum_rotated_side(polygons[index])
                for index in care_indices
            }
            valid_care = {}
            for _ in range(self.crop_tries):
                minimum_width = min(
                    int(width * self.min_crop_side_ratio), self.target_width
                )
                maximum_width = self.target_width * 3
                crop_width = (
                    width
                    if minimum_width >= maximum_width
                    else min(random.randint(max(1, minimum_width), maximum_width), width)
                )
                minimum_height = min(
                    int(height * self.min_crop_side_ratio), self.target_height
                )
                maximum_height = self.target_height * 3
                crop_height = (
                    height
                    if minimum_height >= maximum_height
                    else min(
                        random.randint(max(1, minimum_height), maximum_height),
                        height,
                    )
                )
                crop_x = (
                    0
                    if crop_width >= width
                    else random.randint(0, width - crop_width)
                )
                crop_y = (
                    0
                    if crop_height >= height
                    else random.randint(0, height - crop_height)
                )
                valid_care = {}
                for index in care_indices:
                    polygon = polygons[index]
                    if _is_outside(
                        polygon, crop_x, crop_y, crop_width, crop_height
                    ):
                        continue
                    if _is_inside(polygon, crop_x, crop_y, crop_width, crop_height):
                        valid_care[index] = None
                        continue
                    clipped = _clip_polygon_to_rect(
                        polygon, crop_x, crop_y, crop_width, crop_height
                    )
                    if clipped is None or cv2.contourArea(clipped) < 80:
                        continue
                    character_height = character_heights[index]
                    if _minimum_rotated_side(clipped) < character_height * 0.35:
                        continue
                    if (
                        len(clipped) == 4
                        and _minimum_quad_side(clipped) < character_height * 0.35
                    ):
                        continue
                    valid_care[index] = clipped
                if valid_care:
                    break
            else:
                crop_x, crop_y, crop_width, crop_height = 0, 0, width, height
                valid_care = {index: None for index in care_indices}

        needs_resize = (
            crop_width > self.target_width or crop_height > self.target_height
        )
        scale = (
            min(
                self.target_width / crop_width,
                self.target_height / crop_height,
            )
            if needs_resize
            else 1.0
        )
        resized_width = max(1, int(crop_width * scale))
        resized_height = max(1, int(crop_height * scale))
        crop = image[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
        resized = (
            cv2.resize(crop, (resized_width, resized_height))
            if needs_resize
            else crop
        )
        padding_left = random.randint(0, self.target_width - resized_width)
        padding_top = random.randint(0, self.target_height - resized_height)
        output = np.zeros(
            (self.target_height, self.target_width, 3), dtype=image.dtype
        )
        output[
            padding_top : padding_top + resized_height,
            padding_left : padding_left + resized_width,
        ] = resized

        kept_polygons = []
        kept_ignores = []
        kept_texts = []
        offset = np.asarray([crop_x, crop_y], dtype=np.float32)
        padding = np.asarray([padding_left, padding_top], dtype=np.float32)
        for index, (polygon, ignored, text) in enumerate(
            zip(polygons, ignore_tags, texts)
        ):
            if ignored:
                if _is_outside(
                    polygon, crop_x, crop_y, crop_width, crop_height
                ):
                    continue
                transformed = (polygon - offset) * scale + padding
                transformed[:, 0] = np.clip(
                    transformed[:, 0], 0, self.target_width
                )
                transformed[:, 1] = np.clip(
                    transformed[:, 1], 0, self.target_height
                )
            else:
                if index not in valid_care:
                    continue
                source = valid_care[index]
                transformed = (
                    (polygon if source is None else source) - offset
                ) * scale + padding
            kept_polygons.append(transformed.astype(np.float32))
            kept_ignores.append(ignored)
            kept_texts.append(text)
        return DetectionSample(output, kept_polygons, kept_ignores, kept_texts)

    def __call__(self, sample, external=None):
        if not self.additional_augmentations:
            return self._random_crop(sample)
        sample = self._copy_paste(sample, external)
        image = sample.image
        polygons = [polygon.copy() for polygon in sample.polygons]
        ignore_tags = list(sample.ignore_tags)
        texts = list(sample.texts)
        if random.random() < self.flip_probability:
            image = np.ascontiguousarray(image[:, ::-1])
            width = image.shape[1]
            for polygon in polygons:
                polygon[:, 0] = width - 1 - polygon[:, 0]
        if random.random() < self.rotate_probability:
            image, polygons = _fit_rotation(
                image,
                polygons,
                random.uniform(*self.rotate_range),
            )
        scale = random.uniform(*self.scale_range)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        polygons = [polygon * scale for polygon in polygons]
        return self._random_crop(
            DetectionSample(image, polygons, ignore_tags, texts)
        )


class PaddleRecognitionAugmentation:
    """RecAug and RecConAug behavior used by PP-OCRv5 recognition."""

    def __init__(
        self,
        image_shape,
        max_text_length,
        *,
        concat_probability=0.5,
        external_samples=2,
        tia_probability=0.4,
        crop_probability=0.4,
        reverse_probability=0.4,
        noise_probability=0.4,
        jitter_probability=0.4,
        blur_probability=0.4,
        hsv_probability=0.4,
    ):
        self.image_height = int(image_shape[1])
        self.image_width = int(image_shape[2])
        self.max_text_length = int(max_text_length)
        self.concat_probability = float(concat_probability)
        self.external_samples = int(external_samples)
        self.tia_probability = float(tia_probability)
        self.crop_probability = float(crop_probability)
        self.reverse_probability = float(reverse_probability)
        self.noise_probability = float(noise_probability)
        self.jitter_probability = float(jitter_probability)
        self.blur_probability = float(blur_probability)
        self.hsv_probability = float(hsv_probability)
        self.blur_kernel = cv2.getGaussianKernel(5, 1, cv2.CV_32F)

    def concatenate(self, image, text, external_samples):
        if random.random() > self.concat_probability:
            return image, text
        max_ratio = self.image_width / self.image_height
        for external_image, external_text in external_samples:
            if len(text) + len(external_text) > self.max_text_length:
                break
            combined_ratio = (
                image.shape[1] / image.shape[0]
                + external_image.shape[1] / external_image.shape[0]
            )
            if combined_ratio > max_ratio:
                break
            first_width = max(1, round(image.shape[1] / image.shape[0] * self.image_height))
            second_width = max(
                1,
                round(external_image.shape[1] / external_image.shape[0] * self.image_height),
            )
            image = np.concatenate(
                [
                    cv2.resize(image, (first_width, self.image_height)),
                    cv2.resize(external_image, (second_width, self.image_height)),
                ],
                axis=1,
            )
            text += external_text
        return image, text

    def __call__(self, image):
        height, width = image.shape[:2]
        if random.random() <= self.tia_probability:
            if height >= 20 and width >= 20:
                image = tia_distort(image, random.randint(3, 6))
                image = tia_stretch(image, random.randint(3, 6))
            image = tia_perspective(image)
        height, width = image.shape[:2]
        if random.random() <= self.crop_probability and height >= 20 and width >= 20:
            amount = min(random.randint(1, 8), height - 1)
            image = image[amount:] if random.randint(0, 1) else image[:-amount]
        if random.random() <= self.blur_probability:
            image = cv2.sepFilter2D(image, -1, self.blur_kernel, self.blur_kernel)
        if random.random() <= self.hsv_probability:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            delta = 0.001 * random.random() * (1 if random.random() > 0.5 else -1)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * (1 + delta), 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if random.random() <= self.jitter_probability:
            height, width = image.shape[:2]
            shift = int(random.random() * min(width, height) * 0.01)
            if shift:
                shifted = image.copy()
                image[shift:, shift:] = shifted[:-shift, :-shift]
        if random.random() <= self.noise_probability:
            noise = np.random.normal(0, 0.1**0.5, image.shape)
            image = np.clip(image + 0.5 * noise, 0, 255).astype(np.uint8)
        if random.random() <= self.reverse_probability:
            image = 255 - image
        return image
