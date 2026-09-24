"""Finite shuffled passes over completed-game records, with optional BF16 compute.

Parameters, accumulated parameter gradients, losses, and fresh AdamW state stay
FP32. This is a weights warm start, not optimizer/RNG-state resumption.
"""

import math
import random
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from .losses import training_loss
from .model import ChessModel
from .training_data import TrainingBatch, TrainingExample, collate_examples


def split_game_indices(count: int, *, seed: int, held_out_fraction: float = 0.1):
    """Seeded whole-game split; require both partitions to contain a game."""
    if type(count) is not int or count < 2:
        raise ValueError("need at least two completed games")
    if not math.isfinite(held_out_fraction) or not 0 < held_out_fraction < 1:
        raise ValueError("held_out_fraction must be between zero and one")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    held_out = min(count - 1, max(1, round(count * held_out_fraction)))
    return sorted(indices[held_out:]), sorted(indices[:held_out])


@dataclass(frozen=True)
class MinibatchResult:
    steps: int
    samples_processed: int
    gradient_checks: int
    parameter_max_changes: dict[str, float]
    evaluations: tuple[dict, ...]


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("training stop budget reached at a batch boundary")


def _batch(examples, indices, device):
    return TrainingBatch(
        *(t.to(device) for t in collate_examples([examples[i] for i in indices]))
    )


def _context(device, bf16):
    return torch.autocast(device.type, dtype=torch.bfloat16) if bf16 else nullcontext()


def evaluate_examples(model, examples, *, batch_size, bf16, deadline=None):
    """Position-weighted mean losses; evaluation does not update model weights."""
    if not examples or batch_size <= 0:
        raise ValueError("evaluation needs examples and a positive batch_size")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    totals = {"policy": 0.0, "value": 0.0, "total": 0.0}
    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                _check_deadline(deadline)
                indices = range(start, min(start + batch_size, len(examples)))
                batch = _batch(examples, indices, device)
                with _context(device, bf16):
                    output = model(batch.features)
                # Loss functions explicitly promote logits to FP32; no autocast here.
                loss = training_loss(
                    output, batch.policy_targets, batch.value_targets, batch.legal_mask
                )
                if not torch.isfinite(loss.total).item():
                    raise FloatingPointError("non-finite evaluation loss")
                for name in totals:
                    totals[name] += getattr(loss, name).item() * len(indices)
    finally:
        model.train(was_training)
    return {
        "positions": len(examples),
        **{name: value / len(examples) for name, value in totals.items()},
    }


def train_minibatches(
    model: ChessModel,
    train: Sequence[TrainingExample],
    held_out: Sequence[TrainingExample],
    *,
    epochs: int = 2,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 0,
    bf16: bool = True,
    deadline: float | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> MinibatchResult:
    """Visit each training record once per epoch; include the final partial batch.

    The held-out records are used only in no-grad evaluation. Metrics are measured
    before training and after each epoch. Check finite gradients before every
    optimizer step and finite parameters afterward. These checks synchronize the
    accelerator: this is a diagnostic run, not a peak-throughput benchmark.

    A stop/error leaves earlier updates in memory but returns no successful result.
    The caller must not label such a run complete. Checkpoints are caller-owned.
    """
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not train or not held_out:
        raise ValueError("both train and held_out need examples")
    parameters = dict(model.named_parameters())
    device = next(iter(parameters.values())).device
    if device.type not in ("mps", "cpu"):
        raise ValueError("this diagnostic trainer supports cpu and mps")
    if any(p.dtype != torch.float32 or p.device != device for p in parameters.values()):
        raise ValueError("model parameters must be FP32 on one device")
    _check_deadline(deadline)
    before = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0, foreach=False
    )
    generator = random.Random(seed)
    steps = samples = 0
    evaluations = []

    def measure(epoch):
        for split, examples in (("train", train), ("held_out", held_out)):
            row = {
                "kind": "evaluation",
                "epoch": epoch,
                "step": steps,
                "samples_processed": samples,
                "split": split,
                **evaluate_examples(
                    model, examples, batch_size=batch_size, bf16=bf16, deadline=deadline
                ),
            }
            evaluations.append(row)
            if on_event:
                on_event(row)

    measure(0)
    for epoch in range(1, epochs + 1):
        model.train()
        indices = list(range(len(train)))
        generator.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            _check_deadline(deadline)
            selected = indices[start : start + batch_size]
            batch = _batch(train, selected, device)
            optimizer.zero_grad(set_to_none=True)
            with _context(device, bf16):
                output = model(batch.features)
            if bf16 and any(t.dtype != torch.bfloat16 for t in output):
                raise RuntimeError("BF16 requested but model heads did not return BF16")
            loss = training_loss(
                output, batch.policy_targets, batch.value_targets, batch.legal_mask
            )
            if (
                loss.total.dtype != torch.float32
                or not torch.isfinite(loss.total).item()
            ):
                raise FloatingPointError(f"invalid loss before update {steps + 1}")
            loss.total.backward()
            if any(p.grad is None for p in parameters.values()):
                raise RuntimeError(f"missing gradient before update {steps + 1}")
            if any(p.grad.dtype != torch.float32 for p in parameters.values()):
                raise RuntimeError("parameter gradients must stay FP32")
            if (
                not torch.stack(
                    [torch.isfinite(p.grad).all() for p in parameters.values()]
                )
                .all()
                .item()
            ):
                raise FloatingPointError(
                    f"non-finite gradients before update {steps + 1}"
                )
            optimizer.step()
            if (
                not torch.stack([torch.isfinite(p).all() for p in parameters.values()])
                .all()
                .item()
            ):
                raise FloatingPointError(
                    f"non-finite parameters after update {steps + 1}"
                )
            if steps == 0 and any(
                v.dtype != torch.float32
                for state in optimizer.state.values()
                for v in state.values()
                if isinstance(v, torch.Tensor)
            ):
                raise RuntimeError("optimizer state must stay FP32")
            steps += 1
            samples += len(selected)
            if on_event:
                on_event(
                    {
                        "kind": "update",
                        "epoch": epoch,
                        "step": steps,
                        "batch_size": len(selected),
                        "samples_processed": samples,
                        "policy": loss.policy.item(),
                        "value": loss.value.item(),
                        "total": loss.total.item(),
                    }
                )
        measure(epoch)
    changes = {"trunk": 0.0, "policy_head": 0.0, "value_head": 0.0}
    for name, parameter in parameters.items():
        group = name.split(".", 1)[0]
        changes[group] = max(
            changes[group], (parameter.detach().cpu() - before[name]).abs().max().item()
        )
    return MinibatchResult(steps, samples, steps, changes, tuple(evaluations))
