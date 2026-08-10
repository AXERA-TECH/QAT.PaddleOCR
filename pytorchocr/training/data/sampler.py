"""Recognition multi-scale batch sampler adapted from PytorchOCR."""

from __future__ import annotations

import random

from torch.utils.data import Sampler


class RecognitionMultiScaleBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        *,
        width,
        heights,
        base_height,
        base_batch_size,
        fix_batch_size=False,
        drop_last=True,
        seed=0,
    ):
        self.dataset = dataset
        self.width = int(width)
        self.heights = tuple(int(height) for height in heights)
        self.base_height = int(base_height)
        self.base_batch_size = int(base_batch_size)
        self.fix_batch_size = bool(fix_batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed or 0)
        self.epoch = 0
        if not self.heights or self.base_height not in self.heights:
            raise ValueError("Multi-scale heights must include the base height.")
        if any(height <= 0 or height % 16 for height in self.heights):
            raise ValueError("Recognition multi-scale heights must be positive multiples of 16.")
        if self.width <= 0 or self.width % 8:
            raise ValueError("Recognition multi-scale width must be a positive multiple of 8.")
        if self.base_batch_size <= 0:
            raise ValueError("Recognition multi-scale batch size must be positive.")
        base_elements = self.width * self.base_height * self.base_batch_size
        self.scale_batches = tuple(
            (
                height,
                self.base_batch_size
                if self.fix_batch_size
                else max(1, base_elements // (self.width * height)),
            )
            for height in self.heights
        )
        self.batch_specs = self._build_batch_specs()

    def _build_batch_specs(self):
        specs = []
        consumed = 0
        scale_index = 0
        while consumed < len(self.dataset):
            height, batch_size = self.scale_batches[scale_index % len(self.scale_batches)]
            remaining = len(self.dataset) - consumed
            actual = batch_size if self.drop_last else min(batch_size, remaining)
            specs.append((height, actual))
            consumed += batch_size
            scale_index += 1
        return tuple(specs)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)
        specs = list(self.batch_specs)
        rng.shuffle(specs)
        offset = 0
        for height, batch_size in specs:
            batch = []
            for _ in range(batch_size):
                if offset >= len(indices):
                    if not self.drop_last:
                        break
                    offset = 0
                batch.append((self.width, height, indices[offset]))
                offset += 1
            if batch:
                yield batch

    def __len__(self):
        return len(self.batch_specs)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
