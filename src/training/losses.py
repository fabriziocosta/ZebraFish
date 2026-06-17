from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def prototype_consistency_loss(
    st_logits: torch.Tensor,
    ts_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    temperature_value = float(temperature)
    if temperature_value <= 0.0:
        raise ValueError("temperature must be positive")
    st_scaled = st_logits / temperature_value
    ts_scaled = ts_logits / temperature_value
    st_targets = F.softmax(st_scaled.detach(), dim=-1)
    ts_targets = F.softmax(ts_scaled.detach(), dim=-1)
    st_to_ts = -(ts_targets * F.log_softmax(st_scaled, dim=-1)).sum(dim=-1).mean()
    ts_to_st = -(st_targets * F.log_softmax(ts_scaled, dim=-1)).sum(dim=-1).mean()
    return 0.5 * (st_to_ts + ts_to_st)


def apply_water_vs_other_loss(
    *,
    total_loss: torch.Tensor,
    action_logits: torch.Tensor,
    action_targets: torch.Tensor,
    weight: float,
    water_class_index: int = 0,
) -> tuple[torch.Tensor, float]:
    """Retain the water boundary by collapsing multiclass logits to water vs other."""
    if float(weight) <= 0.0 or action_logits.shape[1] <= 2:
        return total_loss, 0.0
    if not 0 <= int(water_class_index) < action_logits.shape[1]:
        raise ValueError(
            f"water_class_index must be in [0, {action_logits.shape[1]}), "
            f"got {water_class_index}"
        )

    other_indices = [
        index for index in range(action_logits.shape[1])
        if index != int(water_class_index)
    ]
    binary_logits = torch.stack(
        [
            action_logits[:, int(water_class_index)],
            torch.logsumexp(action_logits[:, other_indices], dim=1),
        ],
        dim=1,
    )
    binary_targets = action_targets.ne(int(water_class_index)).to(torch.long)
    binary_loss = F.cross_entropy(binary_logits, binary_targets)
    return total_loss + float(weight) * binary_loss, float(binary_loss.item())


def compute_hierarchical_action_losses(
    *,
    action_logits: torch.Tensor,
    water_logits: torch.Tensor,
    action_targets: torch.Tensor,
    action_criterion: nn.Module,
    water_weight: float,
    water_class_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train water-vs-drug separately from conditional drug action classes."""
    if (
        float(water_weight) <= 0.0
        or action_logits.shape[1] <= 2
        or not 0 <= int(water_class_index) < action_logits.shape[1]
    ):
        action_loss = action_criterion(action_logits, action_targets)
        zero = action_loss.new_tensor(0.0)
        return action_loss, zero, action_loss

    water_targets = action_targets.ne(int(water_class_index)).to(torch.long)
    water_loss = F.cross_entropy(water_logits, water_targets)
    drug_indices = [
        index for index in range(action_logits.shape[1])
        if index != int(water_class_index)
    ]
    drug_mask = action_targets.ne(int(water_class_index))
    if bool(drug_mask.any()):
        drug_index_map = {
            encoded_label: conditional_index
            for conditional_index, encoded_label in enumerate(drug_indices)
        }
        conditional_targets = torch.tensor(
            [drug_index_map[int(target)] for target in action_targets[drug_mask].detach().cpu().tolist()],
            dtype=torch.long,
            device=action_targets.device,
        )
        action_weight = getattr(action_criterion, "weight", None)
        if action_weight is not None:
            action_weight = action_weight[drug_indices]
        action_loss = F.cross_entropy(
            action_logits[drug_mask][:, drug_indices],
            conditional_targets,
            weight=action_weight,
            reduction=getattr(action_criterion, "reduction", "mean"),
            label_smoothing=float(getattr(action_criterion, "label_smoothing", 0.0)),
        )
    else:
        action_loss = action_logits.sum() * 0.0
    total_action_loss = action_loss + float(water_weight) * water_loss
    return action_loss, water_loss, total_action_loss


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
