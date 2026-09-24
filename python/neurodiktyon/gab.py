"""Board-conditioned mixtures of learned square-to-square bias templates."""

from torch import Tensor, nn
from torch.nn import functional as F

from .embedding import SQUARE_COUNT


class GABTemplates(nn.Module):
    """A learned template bank shared across attention heads and GAB layers.

    Input coefficients: [batch, heads, d3]. Output biases: [batch, heads, 64, 64].
    The d3 templates are the columns of projection.weight, reshaped to 64 x 64.
    Coefficients can have either sign and are not normalized into probabilities.
    """

    def __init__(self, d3: int) -> None:
        super().__init__()
        if d3 <= 0:
            raise ValueError("d3 must be positive")
        self.d3 = d3
        self.projection = nn.Linear(d3, SQUARE_COUNT * SQUARE_COUNT, bias=False)

    def forward(self, coefficients: Tensor) -> Tensor:
        if coefficients.ndim != 3 or coefficients.shape[-1] != self.d3:
            raise ValueError(
                f"expected [batch, heads, {self.d3}], got {tuple(coefficients.shape)}"
            )
        return self.projection(coefficients).unflatten(-1, (SQUARE_COUNT, SQUARE_COUNT))


class GeometricAttentionBias(nn.Module):
    """Generate [batch, n_heads, 64, 64] biases from [batch, 64, d_model] tokens.

    Create one GABTemplates object and pass that same object to each layer's
    generator. Only the template bank is shared between layers; each generator
    learns its own board summary and head coefficients. d3 comes from the bank.
    Square order is fixed: flattening makes each square's location significant.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        templates: GABTemplates,
        *,
        d1: int = 8,
        d2: int = 32,
    ) -> None:
        super().__init__()
        if min(d_model, n_heads, d1, d2) <= 0:
            raise ValueError("d_model, n_heads, d1, and d2 must be positive")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d1 = d1
        self.d2 = d2
        self.templates = templates

        self.square_projection = nn.Linear(d_model, d1)
        self.board_projection = nn.Linear(SQUARE_COUNT * d1, d2)
        self.board_norm = nn.LayerNorm(d2)
        self.head_projection = nn.Linear(d2, n_heads * templates.d3)
        self.head_norm = nn.LayerNorm(n_heads * templates.d3)

    def coefficients(self, tokens: Tensor) -> Tensor:
        """Expose the board-dependent [batch, n_heads, d3] template coefficients."""
        if tokens.ndim != 3 or tokens.shape[1:] != (SQUARE_COUNT, self.d_model):
            raise ValueError(
                f"expected [batch, {SQUARE_COUNT}, {self.d_model}], "
                f"got {tuple(tokens.shape)}"
            )

        # Compress per square, then preserve square order in the flattened vector.
        squares = self.square_projection(tokens)  # [B, 64, d1]
        board = squares.flatten(start_dim=1)  # [B, 64*d1]
        summary = self.board_norm(F.gelu(self.board_projection(board)))  # [B, d2]

        # Normalize across ALL heads' coefficients before separating the heads.
        packed = self.head_norm(F.gelu(self.head_projection(summary)))  # [B, H*d3]
        return packed.unflatten(-1, (self.n_heads, self.templates.d3))  # [B, H, d3]

    def forward(self, tokens: Tensor) -> Tensor:
        return self.templates(self.coefficients(tokens))
