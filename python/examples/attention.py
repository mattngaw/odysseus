"""Inspect attention shapes and compare with explicit tensor operations on CPU."""

import argparse
import math

import torch

from neurodiktyon import SQUARE_COUNT, MultiHeadSelfAttention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(0)
    attention = MultiHeadSelfAttention(args.d_model, args.heads)
    batch = 2
    tokens = torch.randn(batch, SQUARE_COUNT, args.d_model)

    with torch.inference_mode():
        packed = attention.qkv(tokens)
        q, k, v = packed.chunk(3, dim=-1)
        q, k, v = (
            tensor.reshape(batch, SQUARE_COUNT, args.heads, attention.d_head).transpose(
                1, 2
            )
            for tensor in (q, k, v)
        )
        # Materialize scores and weights here so the equations are inspectable.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attention.d_head)
        weights = scores.softmax(dim=-1)
        heads = weights @ v
        merged = heads.transpose(1, 2).reshape(batch, SQUARE_COUNT, args.d_model)
        explicit = attention.out_proj(merged)
        output = attention(tokens)
        torch.testing.assert_close(output, explicit, rtol=1e-4, atol=1e-6)

    print("Synthetic token vectors; all-square attention, zero dropout.")
    for name, tensor in [
        ("Input", tokens),
        ("Packed QKV", packed),
        ("Q (K and V have the same shape)", q),
        ("Scaled scores: Q @ K.T / sqrt(d_head)", scores),
        ("Weights: softmax over key squares", weights),
        ("Head outputs: weights @ V", heads),
        ("Concatenated heads", merged),
        ("Output after W_O", output),
    ]:
        print(f"{name:42} {list(tensor.shape)}")
    row_sums = weights.sum(dim=-1)
    print(f"Weight row sums: {row_sums.min():.6f} .. {row_sums.max():.6f}")
    print(f"Trainable parameters: {sum(p.numel() for p in attention.parameters()):,}")
    print(
        f"Explicit equations vs module: max abs error {(output - explicit).abs().max():.3g}"
    )


if __name__ == "__main__":
    main()
