"""Training losses for policy visits and game outcomes."""

import math
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import ModelOutput
from .policy_map import POLICY_SIZE


def policy_loss(logits: Tensor, targets: Tensor, legal_mask: Tensor) -> Tensor:
    """Mean legal-only policy cross-entropy over a nonempty batch.

    All inputs have shape [batch, 1858] on the same device. The boolean mask
    includes every legal move, including zero-visit moves. Targets are already
    normalized visit distributions: finite, nonnegative, zero on illegal moves,
    and summing to one per position (rtol=1e-5, atol=1e-6).

    Each row needs at least one legal move, and its legal logits must be finite.
    Illegal logits are ignored, even if nonfinite, and have zero loss gradient.
    Positions carry equal weight, regardless of their search budgets.
    FP16/BF16 logits are promoted to FP32 for the loss; FP64 logits retain FP64.
    """
    if logits.ndim != 2 or logits.shape[1] != POLICY_SIZE or logits.shape[0] == 0:
        raise ValueError(f"expected nonempty logits [batch, {POLICY_SIZE}]")
    if targets.shape != logits.shape or legal_mask.shape != logits.shape:
        raise ValueError("targets and legal_mask must have the same shape as logits")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("logits and targets must be floating-point tensors")
    if legal_mask.dtype != torch.bool:
        raise TypeError("legal_mask must be boolean")
    if targets.device != logits.device or legal_mask.device != logits.device:
        raise ValueError("logits, targets, and legal_mask must be on the same device")
    if not legal_mask.any(dim=1).all():
        raise ValueError("each position must have at least one legal move")
    if not torch.isfinite(logits[legal_mask]).all():
        raise ValueError("legal logits must be finite")
    if not torch.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("targets must be finite and nonnegative")
    if (targets[~legal_mask] != 0).any():
        raise ValueError("targets must be zero on illegal moves")

    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    targets = targets.to(dtype=dtype)
    sums = targets.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6):
        raise ValueError("targets must sum to one per position")

    masked_logits = logits.to(dtype=dtype).masked_fill(~legal_mask, -torch.inf)
    log_probs = F.log_softmax(masked_logits, dim=1)
    # Illegal targets are zero, but 0 * -inf would still produce NaN.
    log_probs = log_probs.masked_fill(~legal_mask, 0.0)
    return -(targets * log_probs).sum(dim=1).mean()


def value_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean W/D/L cross-entropy over a nonempty batch of recorded positions.

    Both inputs are floating-point tensors of shape [batch, 3] on the same
    device, with columns win, draw, loss. Targets must be one-hot completed-game
    outcomes from each position's side-to-move perspective; the caller supplies
    that perspective. All logits must be finite. Positions carry equal weight.
    FP16/BF16 logits are promoted to FP32 for the loss; FP64 logits retain FP64.
    """
    if logits.ndim != 2 or logits.shape[1] != 3 or logits.shape[0] == 0:
        raise ValueError("expected nonempty logits [batch, 3]")
    if targets.shape != logits.shape:
        raise ValueError("targets must have the same shape as logits")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("logits and targets must be floating-point tensors")
    if targets.device != logits.device:
        raise ValueError("logits and targets must be on the same device")
    if not torch.isfinite(logits).all():
        raise ValueError("value logits must be finite")
    if (
        not ((targets == 0) | (targets == 1)).all()
        or not (targets.sum(dim=1) == 1).all()
    ):
        raise ValueError("targets must be one-hot W/D/L outcomes")

    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    log_probs = F.log_softmax(logits.to(dtype=dtype), dim=1)
    return -(targets.to(dtype=dtype) * log_probs).sum(dim=1).mean()


class TrainingLoss(NamedTuple):
    """Scalar batch means: weighted total and unweighted policy/value components.

    All three tensors retain their autograd graphs. Backpropagate through total;
    use .item() or .detach() when recording metrics.
    """

    total: Tensor
    policy: Tensor
    value: Tensor


def training_loss(
    output: ModelOutput,
    policy_targets: Tensor,
    value_targets: Tensor,
    legal_mask: Tensor,
    *,
    value_weight: float = 1.0,
) -> TrainingLoss:
    """Combine the losses as policy + value_weight * value.

    Inputs follow policy_loss/value_loss contracts and describe the same batch
    of positions in the same order. Both heads must use the same device and
    batch size. Each component already averages over positions; total is not
    averaged again. value_weight is a finite, nonnegative Python scalar.
    A zero weight still validates and reports the unweighted value loss.
    """
    if not isinstance(value_weight, (int, float)):
        raise TypeError("value_weight must be a Python number")
    if not math.isfinite(value_weight) or value_weight < 0:
        raise ValueError("value_weight must be finite and nonnegative")
    if output.policy_logits.device != output.value_logits.device:
        raise ValueError("policy and value logits must be on the same device")
    if output.policy_logits.shape[:1] != output.value_logits.shape[:1]:
        raise ValueError("policy and value logits must have the same batch size")

    policy = policy_loss(output.policy_logits, policy_targets, legal_mask)
    value = value_loss(output.value_logits, value_targets)
    return TrainingLoss(total=policy + value_weight * value, policy=policy, value=value)
