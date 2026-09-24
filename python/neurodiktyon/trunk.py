"""Square embedding and a transformer stack, ready to feed output heads."""

from torch import Tensor, nn

from .block import TransformerBlock
from .embedding import SquareEmbedding
from .gab import GABTemplates


class TransformerTrunk(nn.Module):
    """Map [batch, 64, 110] features to [batch, 64, d_model] square tokens.

    Each block owns its parameters except for one shared GAB template bank.
    A final LayerNorm normalizes each square's features after the residual
    stack. Policy/value heads and input encoding belong outside this module.
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
    ) -> None:
        super().__init__()
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")

        self.d_model = d_model
        self.embedding = SquareEmbedding(d_model)
        self.templates = GABTemplates(gab_d3)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model,
                n_heads,
                d_ff,
                self.templates,
                gab_d1=gab_d1,
                gab_d2=gab_d2,
            )
            for _ in range(n_blocks)
        )
        self.final_norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, features: Tensor) -> Tensor:
        tokens = self.embedding(features)
        for block in self.blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)
