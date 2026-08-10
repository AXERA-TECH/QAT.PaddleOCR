import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import PaddleRecognitionAugmentation


class CTCLabelEncoder:
    def __init__(
        self,
        dictionary_path,
        max_text_length=25,
        use_space_char=False,
    ):
        with open(dictionary_path, encoding="utf-8") as dictionary_stream:
            characters = [line.rstrip("\r\n") for line in dictionary_stream]
        if use_space_char:
            characters.append(" ")
        self.dictionary = {
            character: index + 1 for index, character in enumerate(characters)
        }
        self.max_text_length = max_text_length

    def encode(self, text):
        if not text or len(text) > self.max_text_length:
            return None
        encoded = [self.dictionary[char] for char in text if char in self.dictionary]
        if not encoded:
            return None
        padded = encoded + [0] * (self.max_text_length - len(encoded))
        return np.asarray(padded, dtype=np.int64), len(encoded)


class NRTRLabelEncoder:
    def __init__(
        self,
        dictionary_path,
        max_text_length=25,
        use_space_char=False,
    ):
        with open(dictionary_path, encoding="utf-8") as dictionary_stream:
            characters = [line.rstrip("\r\n") for line in dictionary_stream]
        if use_space_char:
            characters.append(" ")
        self.dictionary = {
            character: index + 4 for index, character in enumerate(characters)
        }
        self.max_text_length = max_text_length

    def encode(self, text):
        if not text or len(text) > self.max_text_length:
            return None
        encoded = [self.dictionary[char] for char in text if char in self.dictionary]
        if not encoded or len(encoded) >= self.max_text_length - 1:
            return None
        sequence = [2, *encoded, 3]
        sequence.extend([0] * (self.max_text_length - len(sequence)))
        return np.asarray(sequence, dtype=np.int64)


def resize_rec_image(image, image_shape):
    channels, target_height, target_width = image_shape
    if channels != 3:
        raise ValueError("Recognition images must have three channels.")
    height, width = image.shape[:2]
    resized_width = min(target_width, int(math.ceil(target_height * width / height)))
    resized = cv2.resize(image, (resized_width, target_height)).astype(np.float32)
    resized = resized.transpose(2, 0, 1) / 255.0
    resized = (resized - 0.5) / 0.5
    output = np.zeros(image_shape, dtype=np.float32)
    output[:, :, :resized_width] = resized
    return output, min(1.0, resized_width / target_width)


class RecognitionDataset(Dataset):
    def __init__(
        self,
        label_file,
        dictionary_path,
        data_dir=".",
        image_shape=(3, 48, 320),
        max_text_length=25,
        use_space_char=False,
        multi_head=False,
        augmentation="none",
    ):
        self.label_file = Path(label_file).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.image_shape = tuple(image_shape)
        self.max_text_length = int(max_text_length)
        self.encoder = CTCLabelEncoder(
            dictionary_path,
            max_text_length=max_text_length,
            use_space_char=use_space_char,
        )
        self.nrtr_encoder = (
            NRTRLabelEncoder(
                dictionary_path,
                max_text_length=max_text_length,
                use_space_char=use_space_char,
            )
            if multi_head
            else None
        )
        if augmentation not in ("none", "paddle"):
            raise ValueError("Recognition augmentation must be 'none' or 'paddle'.")
        self.augmentation = augmentation
        self.augmenter = (
            PaddleRecognitionAugmentation(
                self.image_shape,
                self.max_text_length,
            )
            if augmentation == "paddle"
            else None
        )
        with open(self.label_file, encoding="utf-8") as label_stream:
            raw_samples = [
                line.rstrip("\r\n") for line in label_stream if line.strip()
            ]
        self.samples = []
        for index, line in enumerate(raw_samples):
            try:
                image_name, text = line.split("\t", 1)
            except ValueError as error:
                raise ValueError(
                    f"Invalid recognition label at line {index + 1}"
                ) from error
            encoded = self.encoder.encode(text)
            nrtr_encoded = (
                self.nrtr_encoder.encode(text)
                if self.nrtr_encoder is not None
                else None
            )
            if encoded is not None and (
                self.nrtr_encoder is None or nrtr_encoded is not None
            ):
                self.samples.append((image_name, text, encoded, nrtr_encoded))
        if not self.samples:
            raise ValueError(f"No encodable recognition labels in {self.label_file}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, index):
        image_name = self.samples[index][0]
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = self.data_dir / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not decode recognition image: {image_path}")
        return image

    def __getitem__(self, properties):
        if isinstance(properties, (tuple, list)):
            if len(properties) != 3:
                raise ValueError("Recognition multi-scale index must be (width, height, index).")
            width, height, index = (int(value) for value in properties)
            image_shape = (self.image_shape[0], height, width)
        else:
            index = int(properties)
            image_shape = self.image_shape
        _, text, encoded, nrtr_encoded = self.samples[index]
        image = self._load_image(index)
        if self.augmenter is not None:
            external_samples = []
            for _ in range(self.augmenter.external_samples):
                external_index = np.random.randint(len(self.samples))
                external_samples.append(
                    (self._load_image(external_index), self.samples[external_index][1])
                )
            image, text = self.augmenter.concatenate(image, text, external_samples)
            image = self.augmenter(image)
            encoded = self.encoder.encode(text)
            nrtr_encoded = (
                self.nrtr_encoder.encode(text)
                if self.nrtr_encoder is not None
                else None
            )
            if encoded is None or (
                self.nrtr_encoder is not None and nrtr_encoded is None
            ):
                raise ValueError("Recognition augmentation produced an invalid label.")
        encoded, target_length = encoded
        image, valid_ratio = resize_rec_image(image, image_shape)
        targets = {
            "targets": torch.from_numpy(encoded),
            "target_lengths": torch.tensor(target_length, dtype=torch.long),
            "valid_ratio": torch.tensor(valid_ratio, dtype=torch.float32),
        }
        if nrtr_encoded is not None:
            targets["gtc_targets"] = torch.from_numpy(nrtr_encoded)
        return torch.from_numpy(image), targets
