import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps=1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target, mask):
        intersection = torch.sum(prediction * target * mask)
        union = (
            torch.sum(prediction * mask)
            + torch.sum(target * mask)
            + self.eps
        )
        return 1.0 - 2.0 * intersection / union


class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1.0e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, prediction, target, mask):
        prediction = prediction.clamp(self.eps, 1.0 - self.eps)
        cross_entropy = F.binary_cross_entropy(
            prediction,
            target,
            reduction="none",
        )
        probability = prediction * target + (1.0 - prediction) * (1.0 - target)
        alpha = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal = alpha * (1.0 - probability).pow(self.gamma) * cross_entropy
        return torch.sum(focal * mask) / (torch.sum(mask) + self.eps)


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        dice_weight=1.0,
        focal_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        eps=1.0e-6,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss(eps=eps)
        self.focal = MaskedFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            eps=eps,
        )

    def forward(self, prediction, target, mask):
        return (
            self.dice_weight * self.dice(prediction, target, mask)
            + self.focal_weight * self.focal(prediction, target, mask)
        )


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target, mask):
        return torch.sum(torch.abs(prediction - target) * mask) / (
            torch.sum(mask) + self.eps
        )


class BalanceLoss(nn.Module):
    def __init__(self, balance_loss=True, negative_ratio=3.0, eps=1.0e-6):
        super().__init__()
        self.balance_loss = bool(balance_loss)
        self.negative_ratio = float(negative_ratio)
        self.eps = float(eps)
        self.loss = DiceLoss(eps=eps)

    def forward(self, prediction, target, mask):
        origin_loss = self.loss(prediction, target, mask)
        if not self.balance_loss:
            return origin_loss

        positive = target * mask
        negative = (1.0 - target) * mask
        positive_count = int(positive.detach().sum())
        negative_count = min(
            int(negative.detach().sum()),
            int(positive_count * self.negative_ratio),
        )
        positive_loss = torch.sum(positive * origin_loss)
        if negative_count > 0:
            negative_loss = torch.sort(
                (negative * origin_loss).reshape(-1),
                descending=True,
            ).values[:negative_count].sum()
            return (positive_loss + negative_loss) / (
                positive_count + negative_count + self.eps
            )
        return positive_loss / (positive_count + self.eps)


class DBLoss(nn.Module):
    def __init__(
        self,
        balance_loss=True,
        main_loss_type="DiceFocalLoss",
        alpha=5.0,
        beta=10.0,
        ohem_ratio=3.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        dice_weight=1.0,
        focal_weight=1.0,
        eps=1.0e-6,
        aux_weight_p4=0.0,
        aux_weight_p3=0.0,
        aux_weight_p2=0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        if main_loss_type == "DiceLoss":
            self.segmentation = BalanceLoss(
                balance_loss=balance_loss,
                negative_ratio=ohem_ratio,
                eps=eps,
            )
            self.binary = DiceLoss(eps=eps)
        elif main_loss_type == "DiceFocalLoss":
            self.segmentation = DiceFocalLoss(
                dice_weight=dice_weight,
                focal_weight=focal_weight,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                eps=eps,
            )
            self.binary = self.segmentation
        else:
            raise ValueError(f"Unsupported DB main loss: {main_loss_type}")
        self.threshold = MaskL1Loss(eps=eps)
        self.aux_weights = {
            "aux_maps_p4": float(aux_weight_p4),
            "aux_maps_p3": float(aux_weight_p3),
            "aux_maps_p2": float(aux_weight_p2),
        }

    def forward(self, predictions, labels):
        if isinstance(predictions, dict):
            maps = predictions["maps"]
        elif isinstance(predictions, (tuple, list)):
            maps = torch.cat(tuple(predictions), dim=1)
        else:
            maps = predictions
        if maps.ndim != 4 or maps.shape[1] != 3:
            raise ValueError("DB training maps must have shape [batch, 3, height, width].")
        shrink = maps[:, 0]
        threshold = maps[:, 1]
        binary = maps[:, 2]
        loss_shrink = self.alpha * self.segmentation(
            shrink,
            labels["shrink_map"],
            labels["shrink_mask"],
        )
        loss_threshold = self.beta * self.threshold(
            threshold,
            labels["threshold_map"],
            labels["threshold_mask"],
        )
        loss_binary = self.binary(
            binary,
            labels["shrink_map"],
            labels["shrink_mask"],
        )
        loss_cbn = maps.new_zeros(())
        if isinstance(predictions, dict) and "distance_maps" in predictions:
            loss_cbn = self.segmentation(
                predictions["cbn_maps"][:, 0],
                labels["shrink_map"],
                labels["shrink_mask"],
            )
        losses = {
            "loss": loss_shrink + loss_threshold + loss_binary + loss_cbn,
            "loss_shrink_maps": loss_shrink,
            "loss_threshold_maps": loss_threshold,
            "loss_binary_maps": loss_binary,
            "loss_cbn": loss_cbn,
        }
        if isinstance(predictions, dict):
            for name, weight in self.aux_weights.items():
                if weight <= 0 or name not in predictions:
                    continue
                aux_maps = predictions[name]
                aux_loss = (
                    self.alpha
                    * self.segmentation(
                        aux_maps[:, 0],
                        labels["shrink_map"],
                        labels["shrink_mask"],
                    )
                    + self.beta
                    * self.threshold(
                        aux_maps[:, 1],
                        labels["threshold_map"],
                        labels["threshold_mask"],
                    )
                    + self.binary(
                        aux_maps[:, 2],
                        labels["shrink_map"],
                        labels["shrink_mask"],
                    )
                )
                losses[f"loss_{name}"] = aux_loss
                losses["loss"] = losses["loss"] + weight * aux_loss
        return losses
