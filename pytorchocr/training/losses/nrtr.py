import torch
from torch import nn
from torch.nn import functional as F


class NRTRLoss(nn.Module):
    def __init__(self, smoothing=True, ignore_index=0):
        super().__init__()
        self.smoothing = bool(smoothing)
        self.ignore_index = int(ignore_index)
        self.cross_entropy = nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=self.ignore_index,
        )

    def forward(self, logits, targets):
        if logits.ndim != 3:
            raise ValueError("NRTR logits must have shape [batch, time, classes].")
        labels = targets["gtc_targets"]
        lengths = targets["target_lengths"]
        max_length = int(lengths.max())
        labels = labels[:, 1 : 2 + max_length].long()
        if logits.shape[1] < labels.shape[1]:
            raise ValueError(
                "NRTR logits/target shape mismatch: "
                f"{tuple(logits.shape[:2])} != {tuple(labels.shape)}"
            )
        # The PT2E training graph uses a fixed decoder length for export. The
        # Paddle model dynamically decodes to this batch maximum, so discard
        # only the fixed graph's trailing positions here.
        logits = logits[:, : labels.shape[1], :]
        flat_logits = logits.float().reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        if not self.smoothing:
            loss = self.cross_entropy(flat_logits, flat_labels)
        else:
            epsilon = 0.1
            classes = flat_logits.shape[-1]
            one_hot = F.one_hot(flat_labels, num_classes=classes).to(flat_logits)
            smoothed = one_hot * (1 - epsilon) + (1 - one_hot) * (
                epsilon / (classes - 1)
            )
            per_token = -(smoothed * F.log_softmax(flat_logits, dim=-1)).sum(dim=-1)
            non_padding = flat_labels != self.ignore_index
            if not bool(non_padding.any()):
                raise ValueError("NRTR batch contains no non-padding target tokens.")
            loss = per_token.masked_select(non_padding).mean()
        return {"loss": loss, "loss_nrtr": loss}


class MultiLoss(nn.Module):
    def __init__(self, weight_1=1.0, weight_2=1.0, nrtr_smoothing=True):
        super().__init__()
        from .ctc import CTCLoss

        self.ctc = CTCLoss()
        self.nrtr = NRTRLoss(smoothing=nrtr_smoothing)
        self.ctc_weight = float(weight_1)
        self.nrtr_weight = float(weight_2)

    def forward(self, predictions, targets):
        if torch.is_tensor(predictions):
            ctc = self.ctc(predictions, targets)["loss"] * self.ctc_weight
            return {"CTCLoss": ctc, "loss": ctc}
        if not isinstance(predictions, dict) or not {"ctc", "gtc"}.issubset(predictions):
            raise ValueError("MultiLoss requires 'ctc' and 'gtc' model outputs.")
        ctc = self.ctc(predictions["ctc"], targets)["loss"] * self.ctc_weight
        nrtr = self.nrtr(predictions["gtc"], targets)["loss"] * self.nrtr_weight
        return {
            "CTCLoss": ctc,
            "NRTRLoss": nrtr,
            "loss": ctc + nrtr,
        }
