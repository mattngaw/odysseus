"""Predict side-to-move win/draw/loss logits from contextualized square tokens."""

from torch import Tensor, nn
from torch.nn import functional as F

from .embedding import SQUARE_COUNT


class ValueHead(nn.Module):
    """Map [batch, 64, d_model] tokens to [batch, 3] raw W/D/L logits.

    Mean pooling produces one board summary, followed by LayerNorm and a
    two-layer MLP with ReLU. Both projections have biases; there is no dropout.
    Columns are win, draw, loss from the encoded position's side-to-move view.
    The caller applies softmax for probabilities, then p_win - p_loss for search.
    Training can consume the logits directly with a classification loss.
    """

    def __init__(self, d_model: int, d_hidden: int = 128) -> None:
        super().__init__()
        if d_model <= 0 or d_hidden <= 0:
            raise ValueError("d_model and d_hidden must be positive")

        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.in_proj = nn.Linear(d_model, d_hidden)
        self.out_proj = nn.Linear(d_hidden, 3)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (SQUARE_COUNT, self.d_model):
            raise ValueError(
                f"expected [batch, {SQUARE_COUNT}, {self.d_model}], "
                f"got {tuple(tokens.shape)}"
            )

        summary = self.norm(tokens.mean(dim=1))
        return self.out_proj(F.relu(self.in_proj(summary)))
