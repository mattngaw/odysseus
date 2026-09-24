"""Multi-head self-attention over the 64 square tokens, with optional biases."""

from torch import Tensor, nn
from torch.nn import functional as F

from .embedding import SQUARE_COUNT


class MultiHeadSelfAttention(nn.Module):
    """Map [batch, 64, d_model] to the same shape using all-square attention.

    Q, K, and V share one packed projection, with separate learned weights
    and biases for each. Heads use d_head = d_model / n_heads. Concatenated
    head outputs pass through another learned projection (W_O and its bias).

    Every square can attend to every square, including itself. Optional floating
    attention_bias has exact shape [batch, n_heads, 64, 64] and is added AFTER
    score scaling, BEFORE softmax. It may be supplied by a separate GAB generator.
    Dropout is zero in both training and evaluation. Normalization and residual
    additions belong to the enclosing transformer block.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, tokens: Tensor, attention_bias: Tensor | None = None) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (SQUARE_COUNT, self.d_model):
            raise ValueError(
                f"expected [batch, {SQUARE_COUNT}, {self.d_model}], "
                f"got {tuple(tokens.shape)}"
            )
        batch = tokens.shape[0]
        if attention_bias is not None:
            expected = (batch, self.n_heads, SQUARE_COUNT, SQUARE_COUNT)
            if attention_bias.shape != expected:
                raise ValueError(
                    f"expected attention_bias shape {expected}, "
                    f"got {tuple(attention_bias.shape)}"
                )
            if not attention_bias.is_floating_point():
                raise TypeError("attention_bias must be floating point, not a mask")

        # [B, 64, 3*D] -> [B, 64, 3, H, d_head]. The packed axis is Q, K, V.
        packed = self.qkv(tokens).reshape(
            batch, SQUARE_COUNT, 3, self.n_heads, self.d_head
        )
        q, k, v = packed.unbind(dim=2)
        q = q.transpose(1, 2)  # Each becomes [B, H, 64, d_head].
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # softmax(Q @ K.T / sqrt(d_head) + bias, dim=-1) @ V, per head.
        heads = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_bias, dropout_p=0.0, is_causal=False
        )

        # [B, H, 64, d_head] -> [B, 64, H, d_head] -> [B, 64, D].
        merged = heads.transpose(1, 2).reshape(batch, SQUARE_COUNT, self.d_model)
        return self.out_proj(merged)
