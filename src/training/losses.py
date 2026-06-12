from __future__ import annotations

import torch
from torch import nn


def apply_auxiliary_head_losses(
    *,
    total_loss: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    criterion: nn.Module,
    compound_criterion: nn.Module | None = None,
    concentration_criterion: nn.Module | None = None,
    compound_targets: torch.Tensor | None,
    concentration_targets: torch.Tensor | None,
    compound_weight: float,
    concentration_weight: float,
) -> tuple[torch.Tensor, float, float]:
    compound_loss_value = 0.0
    concentration_loss_value = 0.0
    if compound_targets is not None and "compound_logits" in outputs:
        compound_loss = (compound_criterion or criterion)(outputs["compound_logits"], compound_targets)
        total_loss = total_loss + float(compound_weight) * compound_loss
        compound_loss_value = float(compound_loss.item())
    if concentration_targets is not None and "concentration_logits" in outputs:
        concentration_loss = (concentration_criterion or criterion)(outputs["concentration_logits"], concentration_targets)
        total_loss = total_loss + float(concentration_weight) * concentration_loss
        concentration_loss_value = float(concentration_loss.item())
    return total_loss, compound_loss_value, concentration_loss_value
