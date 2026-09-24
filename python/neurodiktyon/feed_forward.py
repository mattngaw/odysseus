"""Nonlinear feature processing applied independently to each square."""

from torch import Tensor, nn
from torch.nn import functional as F

from .embedding import SQUARE_COUNT


class FeedForward(nn.Module):
    """Apply Linear -> GELU -> Linear with shared weights at every square.

    Input and output: [batch, 64, d_model]. The intermediate width is d_ff.
    Both projections have biases. There is no dropout; normalization and
    residual additions belong to the enclosing transformer block.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")

        self.d_model = d_model
        self.d_ff = d_ff
        self.in_proj = nn.Linear(d_model, d_ff)
        self.out_proj = nn.Linear(d_ff, d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (SQUARE_COUNT, self.d_model):
            raise ValueError(
                f"expected [batch, {SQUARE_COUNT}, {self.d_model}], "
                f"got {tuple(tokens.shape)}"
            )
        return self.out_proj(F.gelu(self.in_proj(tokens)))
