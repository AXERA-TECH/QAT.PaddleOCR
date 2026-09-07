import string

import torch

from pytorchocr.postprocess.rec_postprocess import CTCLabelDecode


def levenshtein_distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


class CTCRecognitionMetric:
    def __init__(
        self,
        dictionary_path,
        use_space_char=False,
        main_indicator="acc",
        ignore_space=True,
        is_filter=False,
    ):
        self.decoder = CTCLabelDecode(
            character_dict_path=str(dictionary_path),
            use_space_char=bool(use_space_char),
        )
        self.main_indicator = str(main_indicator)
        if self.main_indicator not in {"acc", "norm_edit_dis"}:
            raise ValueError(
                f"Unsupported recognition main indicator: {self.main_indicator}"
            )
        self.ignore_space = bool(ignore_space)
        self.is_filter = bool(is_filter)
        self.reset()

    def _decode(self, indices, remove_duplicates):
        texts = []
        characters = self.decoder.character
        ignored = set(self.decoder.get_ignored_tokens())
        for sequence in indices:
            decoded = []
            previous = None
            for raw_index in sequence:
                index = int(raw_index)
                duplicate = remove_duplicates and previous == index
                previous = index
                if duplicate or index in ignored:
                    continue
                if not 0 <= index < len(characters):
                    raise ValueError(f"CTC class index {index} is outside the dictionary.")
                decoded.append(characters[index])
            texts.append("".join(decoded))
        return texts

    def _normalize(self, text):
        if self.ignore_space:
            text = text.replace(" ", "")
        if self.is_filter:
            allowed = string.digits + string.ascii_letters
            text = "".join(character for character in text if character in allowed)
            text = text.lower()
        return text

    def update(self, outputs, targets):
        if isinstance(outputs, dict):
            outputs = outputs["ctc"]
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if not torch.is_tensor(outputs) or outputs.ndim != 3:
            raise ValueError("Recognition metric expects [batch, time, classes] logits.")
        labels = targets.get("targets")
        if not torch.is_tensor(labels) or labels.ndim != 2:
            raise ValueError("Recognition metric requires padded 2D CTC targets.")
        prediction_indices = outputs.detach().argmax(dim=-1).cpu().numpy()
        label_indices = labels.detach().cpu().numpy()
        predictions = self._decode(prediction_indices, remove_duplicates=True)
        references = self._decode(label_indices, remove_duplicates=False)
        for prediction, reference in zip(predictions, references):
            prediction = self._normalize(prediction)
            reference = self._normalize(reference)
            self.correct += int(prediction == reference)
            denominator = max(len(prediction), len(reference))
            if denominator:
                self.normalized_distance += (
                    levenshtein_distance(prediction, reference) / denominator
                )
            self.samples += 1

    def compute(self):
        if self.samples == 0:
            raise ValueError("Recognition metric has no samples.")
        return {
            "acc": self.correct / self.samples,
            "norm_edit_dis": 1.0 - self.normalized_distance / self.samples,
        }

    def reset(self):
        self.correct = 0
        self.normalized_distance = 0.0
        self.samples = 0
