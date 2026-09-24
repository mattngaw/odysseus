"""Inspect promotion offsets and Q/R/B logits using shared destination vectors."""

import argparse

import torch

from neurodiktyon import SQUARE_COUNT, PolicyPairScorer, PromotionScorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-policy", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(0)
    pairs = PolicyPairScorer(args.d_model, args.d_policy)
    promotions = PromotionScorer(args.d_policy)
    tokens = torch.randn(2, SQUARE_COUNT, args.d_model)
    with torch.inference_mode():
        pair_logits, destinations = pairs.forward_with_destinations(tokens)
        rank8 = destinations[:, 56:64]
        raw = promotions.projection(rank8)
        offsets = raw[..., :3] + raw[..., 3:4]
        promotion_logits = promotions(pair_logits, destinations)
        expected = pair_logits[:, 48:56, 56:64, None] + offsets[:, None]
        torch.testing.assert_close(promotion_logits, expected)

    print("Synthetic tokens and random weights; no trained chess predictions.")
    for name, tensor in [
        ("Base pair logits", pair_logits),
        ("Reused destination vectors (K)", destinations),
        ("Relative rank-8 destinations", rank8),
        ("Raw Q / R / B / common adjustments", raw),
        ("Q/R/B offsets after adding common", offsets),
        ("Promotion logits [batch, from file, to file, Q/R/B]", promotion_logits),
    ]:
        print(f"{name:51} {list(tensor.shape)}")
    print(
        f"Added promotion parameters: {sum(p.numel() for p in promotions.parameters()):,}"
    )
    print(f"Batch 0, e8 raw [Q, R, B, common]: {raw[0, 4].tolist()}")
    print(f"Batch 0, e8 combined [Q, R, B] offsets: {offsets[0, 4].tolist()}")
    print("Relative route    knight/base       queen        rook      bishop")
    for label, source_file in [("d7 -> e8", 3), ("e7 -> e8", 4)]:
        knight = pair_logits[0, 48 + source_file, 60].item()
        queen, rook, bishop = promotion_logits[0, source_file, 4].tolist()
        print(
            f"{label:15} {knight:+12.6f} {queen:+11.6f} {rook:+11.6f} {bishop:+11.6f}"
        )
    print("Both routes use the same e8 promotion offsets and their own base score.")
    print(
        "All 8 x 8 file pairs are present; selecting the 66 extra vocabulary slots comes later."
    )


if __name__ == "__main__":
    main()
