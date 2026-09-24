"""Shared transformer trunk with policy and value output heads."""

from dataclasses import asdict, dataclass
from typing import NamedTuple

from torch import Tensor, nn

from .policy import PolicyHead
from .trunk import TransformerTrunk
from .value import ValueHead


@dataclass(frozen=True)
class ModelConfig:
    """Complete architecture description saved with model snapshots."""

    n_blocks: int
    d_model: int
    n_heads: int
    d_ff: int
    gab_d1: int
    gab_d2: int
    gab_d3: int
    d_policy: int
    d_value_hidden: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("model dimensions must be positive integers")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


class ModelOutput(NamedTuple):
    """Raw logits: policy [batch, 1858] and side-to-move W/D/L [batch, 3]."""

    policy_logits: Tensor
    value_logits: Tensor


class ChessModel(nn.Module):
    """Predict policy and value logits from [batch, 64, 110] features.

    Inputs follow Pyxis's current-player encoding. Both heads consume the same
    trunk output. Input encoding, legal masking, and softmax remain external.
    """

    def __init__(
        self,
        *,
        n_blocks: int = 8,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 256,
        gab_d1: int = 8,
        gab_d2: int = 32,
        gab_d3: int = 32,
        d_policy: int = 256,
        d_value_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.config = ModelConfig(
            n_blocks,
            d_model,
            n_heads,
            d_ff,
            gab_d1,
            gab_d2,
            gab_d3,
            d_policy,
            d_value_hidden,
        )
        self.trunk = TransformerTrunk(
            n_blocks=n_blocks,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            gab_d1=gab_d1,
            gab_d2=gab_d2,
            gab_d3=gab_d3,
        )
        self.policy_head = PolicyHead(d_model, d_policy)
        self.value_head = ValueHead(d_model, d_value_hidden)

    def forward(self, features: Tensor) -> ModelOutput:
        tokens = self.trunk(features)
        return ModelOutput(
            policy_logits=self.policy_head(tokens),
            value_logits=self.value_head(tokens),
        )
