"""Versioned FP32 model snapshots, portable between CPU and accelerators.

These contain model weights and architecture/encoding metadata, not optimizer
or RNG state. Loading returns an evaluation model, not a resumed training run.
"""

from dataclasses import asdict
from os import PathLike
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor

from .model import ChessModel, ModelConfig
from .training_data import INPUT_ENCODING, POLICY_VOCABULARY

_HEADER = {
    "format": "neurodiktyon.model",
    "version": 1,
    "architecture": "chess_model_v1",
    "input_encoding": INPUT_ENCODING,
    "policy_vocabulary": POLICY_VOCABULARY,
    "value_order": ["win", "draw", "loss"],
    "dtype": "float32",
}


class LoadedCheckpoint(NamedTuple):
    model: ChessModel
    step: int


def _new_model(config: ModelConfig) -> ChessModel:
    # Construction initializes weights that will be replaced; don't perturb the
    # caller's CPU RNG just to inspect or load a snapshot.
    with torch.random.fork_rng(devices=[]):
        return ChessModel(**asdict(config)).float()


def _validate_state(model: ChessModel, state: dict[str, Tensor]) -> None:
    expected = model.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("checkpoint state keys disagree with model configuration")
    for name, reference in expected.items():
        tensor = state[name]
        if (
            not isinstance(tensor, Tensor)
            or tensor.layout != torch.strided
            or tensor.shape != reference.shape
            or tensor.dtype != reference.dtype
        ):
            raise ValueError(f"checkpoint shape/dtype mismatch: {name}")
        if tensor.is_floating_point():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"non-finite checkpoint tensor: {name}")
        elif not torch.equal(tensor, reference):
            # Policy gather buffers are fixed vocabulary data, not learned weights.
            raise ValueError(f"checkpoint vocabulary buffer mismatch: {name}")
    shared = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        previous = shared.setdefault(id(parameter), name)
        if not torch.equal(state[name], state[previous]):
            raise ValueError(f"inconsistent shared parameter: {name} and {previous}")


def save_checkpoint(path: str | PathLike[str], model: ChessModel, *, step: int) -> None:
    """Write a new model snapshot; refuse overwrites and incompatible state.

    All tensors are stored on CPU. Validation happens before file creation.
    On I/O failure a partial file may remain and must not be treated as complete.
    """
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    _validate_state(_new_model(model.config), state)
    payload = {
        **_HEADER,
        "model_config": asdict(model.config),
        "step": step,
        "model_state": state,
    }
    with Path(path).open("xb") as target:
        torch.save(payload, target)


def load_checkpoint(
    path: str | PathLike[str], *, device: str | torch.device = "cpu"
) -> LoadedCheckpoint:
    """Load compatible weights strictly, including fixed and shared parameters."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_keys = _HEADER.keys() | {"model_config", "step", "model_state"}
    if not isinstance(payload, dict) or payload.keys() != expected_keys:
        raise ValueError("invalid model checkpoint fields")
    if type(payload["version"]) is not int or any(
        payload[name] != value for name, value in _HEADER.items()
    ):
        raise ValueError("unsupported checkpoint version, architecture, or encoding")
    if type(payload["step"]) is not int or payload["step"] < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    if not isinstance(payload["model_config"], dict):
        raise ValueError("invalid model configuration")
    try:
        config = ModelConfig(**payload["model_config"])
    except TypeError as error:
        raise ValueError("invalid model configuration fields") from error
    model = _new_model(config)
    _validate_state(model, payload["model_state"])
    model.load_state_dict(payload["model_state"], strict=True)
    return LoadedCheckpoint(model.to(device).eval(), payload["step"])
