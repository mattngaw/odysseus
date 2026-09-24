"""A deliberately small fixed-batch optimizer loop for plumbing/overfit checks."""

import math
from dataclasses import dataclass

import torch

from .losses import training_loss
from .model import ChessModel
from .training_data import TrainingBatch


@dataclass(frozen=True)
class LossRecord:
    step: int  # Number of updates completed before this measurement.
    policy: float
    value: float
    total: float


@dataclass(frozen=True)
class FixedBatchResult:
    losses: tuple[LossRecord, ...]
    gradient_checks: int
    parameter_max_changes: dict[str, float]


def train_fixed_batch(
    model: ChessModel,
    batch: TrainingBatch,
    *,
    steps: int,
    learning_rate: float = 1e-3,
) -> FixedBatchResult:
    """Mutate model with FP32 AdamW, equal policy/value weight, no weight decay.

    Model and batch must already share a device. Every update checks that all
    parameter gradients exist and are finite, and that updated weights remain
    finite. These synchronizing diagnostics make this unsuitable as a throughput
    benchmark. An error stops before the next update; prior updates remain.
    The optimizer is local: this function does not support training resumption.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if (
        type(learning_rate) not in (int, float)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    parameters = dict(model.named_parameters())
    device = next(iter(parameters.values())).device
    if any(p.dtype != torch.float32 or p.device != device for p in parameters.values()):
        raise ValueError("model parameters must be FP32 on one device")
    if any(tensor.device != device for tensor in batch):
        raise ValueError("model and all batch tensors must share a device")
    if (
        any(
            t.dtype != torch.float32
            for t in (
                batch.features,
                batch.policy_targets,
                batch.value_targets,
            )
        )
        or batch.legal_mask.dtype != torch.bool
    ):
        raise ValueError("batch must use FP32 features/targets and a boolean mask")
    before = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    history = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(step < steps):
            losses = training_loss(
                model(batch.features),
                batch.policy_targets,
                batch.value_targets,
                batch.legal_mask,
            )
        if not torch.isfinite(losses.total).item():
            raise FloatingPointError(f"non-finite loss at step {step}")
        history.append(
            LossRecord(
                step, losses.policy.item(), losses.value.item(), losses.total.item()
            )
        )
        if step == steps:
            break
        losses.total.backward()
        missing = [name for name, p in parameters.items() if p.grad is None]
        if missing:
            raise RuntimeError(f"missing gradients at step {step}: {missing}")
        if (
            not torch.stack([torch.isfinite(p.grad).all() for p in parameters.values()])
            .all()
            .item()
        ):
            raise FloatingPointError(f"non-finite gradients at step {step}")
        optimizer.step()
        if (
            not torch.stack([torch.isfinite(p).all() for p in parameters.values()])
            .all()
            .item()
        ):
            raise FloatingPointError(f"non-finite parameters after step {step + 1}")
    changes = {"trunk": 0.0, "policy_head": 0.0, "value_head": 0.0}
    for name, parameter in parameters.items():
        group = name.split(".", 1)[0]
        change = (parameter.detach().cpu() - before[name]).abs().max().item()
        changes[group] = max(changes[group], change)
    return FixedBatchResult(tuple(history), steps, changes)
