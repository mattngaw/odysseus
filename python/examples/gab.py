"""Inspect GAB shapes, template mixtures, and their effect on attention on CPU."""

import torch
from torch import nn
from torch.nn import functional as F

from neurodiktyon import GABTemplates, GeometricAttentionBias, MultiHeadSelfAttention


def main() -> None:
    torch.manual_seed(0)
    templates = GABTemplates(d3=32)
    gab = GeometricAttentionBias(256, 8, templates, d1=8, d2=32)
    attention = MultiHeadSelfAttention(256, 8)
    tokens = torch.randn(2, 64, 256)

    with torch.inference_mode():
        squares = gab.square_projection(tokens)
        flattened = squares.flatten(start_dim=1)
        summary = gab.board_norm(F.gelu(gab.board_projection(flattened)))
        packed = gab.head_norm(F.gelu(gab.head_projection(summary)))
        coefficients = gab.coefficients(tokens)
        biases = gab(tokens)
        # Each column of the final projection is one 64 x 64 template.
        patterns = templates.projection.weight.T.reshape(32, 64, 64)
        explicit = sum(
            coefficients[..., r, None, None] * patterns[r] for r in range(32)
        )
        torch.testing.assert_close(biases, explicit, rtol=1e-4, atol=1e-6)
        plain = attention(tokens)
        biased = attention(tokens, attention_bias=biases)

    print("Synthetic tokens; random initial weights, not learned chess patterns.")
    for name, tensor in [
        ("Tokens", tokens),
        ("Per-square compression", squares),
        ("Flattened board", flattened),
        ("Board summary after GELU + LayerNorm", summary),
        ("Packed head coefficients after GELU + LayerNorm", packed),
        ("Coefficients grouped by head", coefficients),
        ("Shared templates", patterns),
        ("Generated attention biases", biases),
        ("Attention output with GAB", biased),
    ]:
        print(f"{name:49} {list(tensor.shape)}")
    print(f"Template mixture max abs error: {(biases - explicit).abs().max():.3g}")
    print(f"Attention output change with GAB: {(biased - plain).abs().max():.6f}")

    second_gab = GeometricAttentionBias(256, 8, templates, d1=8, d2=32)
    generators = nn.ModuleList([gab, second_gab])
    shared = sum(p.numel() for p in templates.parameters())
    per_layer = sum(p.numel() for p in gab.parameters()) - shared
    print(f"Shared template parameters: {shared:,}")
    print(f"Generator parameters per layer: {per_layer:,}")
    print(
        f"Two generators reference the same bank: {gab.templates is second_gab.templates}"
    )
    print(
        f"Unique parameters for both: {sum(p.numel() for p in generators.parameters()):,}"
    )


if __name__ == "__main__":
    main()
