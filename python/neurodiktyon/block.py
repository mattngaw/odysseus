"""One pre-LayerNorm transformer block over the 64 square tokens."""

from torch import Tensor, nn

from .attention import MultiHeadSelfAttention
from .embedding import SQUARE_COUNT
from .feed_forward import FeedForward
from .gab import GABTemplates, GeometricAttentionBias


class TransformerBlock(nn.Module):
    """Preserve [batch, 64, d_model] through attention and FFN residual branches.

    Attention and GAB receive the same normalized tokens. The FFN receives a
    separately normalized view of the updated residual stream. Both LayerNorms
    normalize each square's features with learned scale/bias and epsilon 1e-5.
    Residual additions have scale one; there is no dropout or output norm.

    Pass the same GABTemplates object to every block. All other parameters,
    including GAB's coefficient generator and internal norms, belong to a block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        templates: GABTemplates,
        *,
        gab_d1: int = 8,
        gab_d2: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.gab = GeometricAttentionBias(
            d_model, n_heads, templates, d1=gab_d1, d2=gab_d2
        )
        self.norm_attention = nn.LayerNorm(d_model, eps=1e-5)
        self.norm_ffn = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (SQUARE_COUNT, self.d_model):
            raise ValueError(
                f"expected [batch, {SQUARE_COUNT}, {self.d_model}], "
                f"got {tuple(tokens.shape)}"
            )

        normalized = self.norm_attention(tokens)
        after_attention = tokens + self.attention(
            normalized, attention_bias=self.gab(normalized)
        )
        return after_attention + self.ffn(self.norm_ffn(after_attention))
