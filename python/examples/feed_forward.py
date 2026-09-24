"""Inspect the per-square FFN using synthetic tokens on CPU."""

import argparse

import torch
from torch.nn import functional as F

from neurodiktyon import SQUARE_COUNT, FeedForward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(0)
    ffn = FeedForward(args.d_model, args.d_ff)
    tokens = torch.randn(2, SQUARE_COUNT, args.d_model)
    tokens[0, 9] = tokens[0, 8]  # Identical inputs at a2 and b2.
    changed = tokens.clone()
    changed[0, 8, 0] += 1.0

    with torch.inference_mode():
        projected = ffn.in_proj(tokens)
        activated = F.gelu(projected)
        output = ffn(tokens)
        changed_output = ffn(changed)

    print("Synthetic tokens and randomly initialized weights; no trained chess model.")
    print(f"Input tokens:      {list(tokens.shape)}")
    print(f"First projection: {list(projected.shape)}")
    print(f"GELU:             {list(activated.shape)}")
    print(f"Output tokens:    {list(output.shape)}")
    print(f"Trainable parameters: {sum(p.numel() for p in ffn.parameters()):,}")
    print(
        f"Identical a2/b2 inputs give identical outputs: {torch.equal(output[0, 8], output[0, 9])}"
    )
    affected = (output != changed_output).any(dim=-1).nonzero().tolist()
    print(f"Changing a2 affects only these [batch, square] indices: {affected}")


if __name__ == "__main__":
    main()
