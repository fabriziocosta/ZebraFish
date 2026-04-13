from __future__ import annotations

import torch
from torch import nn


def commutative_consistency_loss(
    st_logits: torch.Tensor,
    ts_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    st_targets = torch.softmax(st_logits.detach() / temperature, dim=1)
    ts_targets = torch.softmax(ts_logits.detach() / temperature, dim=1)
    st_log_probs = torch.log_softmax(st_logits / temperature, dim=1)
    ts_log_probs = torch.log_softmax(ts_logits / temperature, dim=1)
    loss_st = -(st_targets * ts_log_probs).sum(dim=1).mean()
    loss_ts = -(ts_targets * st_log_probs).sum(dim=1).mean()
    return 0.5 * (loss_st + loss_ts)


def apply_auxiliary_head_losses(
    *,
    total_loss: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    criterion: nn.Module,
    compound_targets: torch.Tensor | None,
    concentration_targets: torch.Tensor | None,
    compound_weight: float,
    concentration_weight: float,
) -> tuple[torch.Tensor, float, float]:
    compound_loss_value = 0.0
    concentration_loss_value = 0.0
    if compound_targets is not None and "compound_logits" in outputs:
        compound_loss = criterion(outputs["compound_logits"], compound_targets)
        total_loss = total_loss + float(compound_weight) * compound_loss
        compound_loss_value = float(compound_loss.item())
    if concentration_targets is not None and "concentration_logits" in outputs:
        concentration_loss = criterion(outputs["concentration_logits"], concentration_targets)
        total_loss = total_loss + float(concentration_weight) * concentration_loss
        concentration_loss_value = float(concentration_loss.item())
    return total_loss, compound_loss_value, concentration_loss_value
