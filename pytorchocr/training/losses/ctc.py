import torch
from torch import nn
from torch.nn import functional as F


class CTCLoss(nn.Module):
    def __init__(self, blank=0, zero_infinity=True):
        super().__init__()
        self.loss = nn.CTCLoss(
            blank=blank,
            reduction="none",
            zero_infinity=zero_infinity,
        )

    def forward(self, logits, targets, target_lengths=None, input_lengths=None):
        if logits.ndim != 3:
            raise ValueError("CTC logits must have shape [batch, time, classes].")
        if isinstance(targets, dict):
            target_lengths = targets["target_lengths"]
            input_lengths = targets.get("input_lengths", input_lengths)
            targets = targets["targets"]
        if target_lengths is None:
            raise ValueError("CTC target_lengths are required.")
        batch_size, time_steps, _ = logits.shape
        if input_lengths is None:
            input_lengths = torch.full(
                (batch_size,),
                time_steps,
                dtype=torch.long,
                device=logits.device,
            )
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        per_sample_loss = self.loss(
            log_probs,
            targets.long(),
            input_lengths.long(),
            target_lengths.long(),
        )
        # PaddleOCR applies CTCLoss(reduction="none") followed by a batch mean.
        # PyTorch's built-in "mean" additionally divides by each target length.
        loss = per_sample_loss.mean()
        return {"loss": loss, "loss_ctc": loss}
