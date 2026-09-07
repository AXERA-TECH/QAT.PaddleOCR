"""PaddleOCR-compatible training augmentations adapted from PytorchOCR."""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np

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


def _split_regions(axis):
    if len(axis) == 0:
        return []
    split_at = np.where(np.diff(axis) != 1)[0] + 1
    return [region for region in np.split(axis, split_at) if len(region)]


def _select_crop_axis(axis, regions, max_size):
    if len(regions) > 1:
        selected = random.choices(regions, k=2)
        values = [int(random.choice(region)) for region in selected]
    else:
        values = random.choices(axis.tolist(), k=2)
    return max(0, min(values)), min(max_size - 1, max(values))


def _crop_area(image, polygons, min_side_ratio, max_tries):
    height, width = image.shape[:2]
    occupied_x = np.zeros(width, dtype=np.uint8)
    occupied_y = np.zeros(height, dtype=np.uint8)
    for polygon in polygons:
        points = np.round(polygon).astype(np.int32)
        xmin = int(np.clip(points[:, 0].min(), 0, width - 1))
        xmax = int(np.clip(points[:, 0].max(), 0, width - 1))
        ymin = int(np.clip(points[:, 1].min(), 0, height - 1))
        ymax = int(np.clip(points[:, 1].max(), 0, height - 1))
        occupied_x[xmin:xmax] = 1
        occupied_y[ymin:ymax] = 1
    free_x = np.where(occupied_x == 0)[0]
    free_y = np.where(occupied_y == 0)[0]
    if len(free_x) == 0 or len(free_y) == 0:
        return 0, 0, width, height
    regions_x = _split_regions(free_x)
    regions_y = _split_regions(free_y)
    for _ in range(max_tries):
        xmin, xmax = _select_crop_axis(free_x, regions_x, width)
        ymin, ymax = _select_crop_axis(free_y, regions_y, height)
        crop_width = xmax - xmin
        crop_height = ymax - ymin
        if crop_width < min_side_ratio * width or crop_height < min_side_ratio * height:
            continue
        if any(
            not _is_outside(polygon, xmin, ymin, crop_width, crop_height)
            for polygon in polygons
        ):
            return xmin, ymin, crop_width, crop_height
    return 0, 0, width, height


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
        copy_paste_ratio=0.2,
        flip_probability=0.5,
        rotate_range=(-10.0, 10.0),
        scale_range=(0.5, 3.0),
        crop_tries=50,
        min_crop_side_ratio=0.1,
    ):
        self.target_height = int(image_shape[1])
        self.target_width = int(image_shape[2])
        self.copy_paste_ratio = float(copy_paste_ratio)
        self.flip_probability = float(flip_probability)
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

    def __call__(self, sample, external=None):
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
        image, polygons = _fit_rotation(
            image,
            polygons,
            random.uniform(*self.rotate_range),
        )
        scale = random.uniform(*self.scale_range)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        polygons = [polygon * scale for polygon in polygons]
        care_polygons = [
            polygon for polygon, ignored in zip(polygons, ignore_tags) if not ignored
        ]
        crop_x, crop_y, crop_width, crop_height = _crop_area(
            image,
            care_polygons,
            self.min_crop_side_ratio,
            self.crop_tries,
        )
        scale = min(self.target_width / crop_width, self.target_height / crop_height)
        resized_width = max(1, int(round(crop_width * scale)))
        resized_height = max(1, int(round(crop_height * scale)))
        crop = image[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
        resized = cv2.resize(crop, (resized_width, resized_height))
        output = np.zeros((self.target_height, self.target_width, 3), dtype=image.dtype)
        output[:resized_height, :resized_width] = resized
        kept_polygons = []
        kept_ignores = []
        kept_texts = []
        offset = np.asarray([crop_x, crop_y], dtype=np.float32)
        for polygon, ignored, text in zip(polygons, ignore_tags, texts):
            transformed = (polygon - offset) * scale
            if _is_outside(transformed, 0, 0, resized_width, resized_height):
                continue
            kept_polygons.append(transformed)
            kept_ignores.append(ignored)
            kept_texts.append(text)
        return DetectionSample(output, kept_polygons, kept_ignores, kept_texts)


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
