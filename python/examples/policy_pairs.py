"""Inspect source/destination projections and raw pair logits on CPU."""

import argparse
import math

import torch

from neurodiktyon import SQUARE_COUNT, PolicyPairScorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-policy", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(0)
    scorer = PolicyPairScorer(args.d_model, args.d_policy)
    tokens = torch.randn(2, SQUARE_COUNT, args.d_model)
    with torch.inference_mode():
        sources = scorer.from_proj(tokens)
        destinations = scorer.to_proj(tokens)
        logits = scorer(tokens)
        # Relative square order a1=0, ..., h8=63 gives e2=12 and e4=28.
        dot = torch.dot(sources[0, 12], destinations[0, 28])
        explicit = dot / math.sqrt(args.d_policy)
        torch.testing.assert_close(logits[0, 12, 28], explicit, rtol=1e-4, atol=1e-6)

    print("Synthetic tokens and random weights; no trained chess predictions.")
    for name, tensor in [
        ("Input square tokens", tokens),
        ("Source projection (Q)", sources),
        ("Destination projection (K)", destinations),
        ("Raw pair logits [batch, from, to]", logits),
    ]:
        print(f"{name:35} {list(tensor.shape)}")
    print(f"Scale: 1 / sqrt({args.d_policy}) = {scorer.scale:.6f}")
    print(f"Trainable parameters: {sum(p.numel() for p in scorer.parameters()):,}")
    print("Batch 0, relative coordinates:")
    print(
        f"  e2 -> e4: {dot.item():+.6f} / sqrt({args.d_policy}) = {explicit.item():+.6f}"
    )
    print(f"  e4 -> e2: {logits[0, 28, 12].item():+.6f} (a separate directed score)")
    print(f"  e2 -> e2: {logits[0, 12, 12].item():+.6f} (computed, but not a move)")
    print(
        "These 4,096 scores precede promotion handling and the 1,858-slot vocabulary."
    )
    print("Legal-move selection and softmax happen later; these are raw logits.")


if __name__ == "__main__":
    main()
