"""Inspect a shared square projection using synthetic feature probes on CPU."""

import argparse

import torch

from neurodiktyon import FEATURE_COUNT, SQUARE_COUNT, SquareEmbedding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(0)
    embedding = SquareEmbedding(args.d_model)
    features = torch.zeros(1, SQUARE_COUNT, FEATURE_COUNT)
    features[0, 8:10, 0] = 1.0  # Two identical "our pawn" probes at a2 and b2.

    with torch.inference_mode():
        tokens = embedding(features)

    print("Synthetic feature probes, not a complete encoded chess position.")
    print(f"Input: {list(features.shape)} -> output: {list(tokens.shape)}")
    print(f"Trainable parameters: {sum(p.numel() for p in embedding.parameters()):,}")
    print("Same weight matrix and bias at every square; no positional information yet.")
    print(
        f"a2 and b2 embeddings are identical: {torch.equal(tokens[0, 8], tokens[0, 9])}"
    )
    print(f"a2 embedding (first 8 components): {tokens[0, 8, :8].tolist()}")


if __name__ == "__main__":
    main()
