"""Inspect the fixed vocabulary gather using identifiable synthetic logits."""

import torch

from neurodiktyon import BASE_MOVE_COUNT, POLICY_SIZE, PolicyMap


def main() -> None:
    mapping = PolicyMap()
    # Values identify each flattened source cell, rather than model predictions.
    pairs = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 64, 64)
    promotions = (10_000 + torch.arange(8 * 8 * 3, dtype=torch.float32)).reshape(
        1, 8, 8, 3
    )
    logits = mapping(pairs, promotions)
    print("Synthetic cell identifiers; these are not learned chess scores.")
    print(f"Pair logits: {list(pairs.shape)} -> {BASE_MOVE_COUNT} base entries")
    print(
        f"Promotion logits: {list(promotions.shape)} -> {POLICY_SIZE - BASE_MOVE_COUNT} entries"
    )
    print(f"Output in Pyxis vocabulary order: {list(logits.shape)}")
    print(f"Learned parameters: {sum(p.numel() for p in mapping.parameters())}")
    print("Index  Relative entry  Gathered cell identifier")
    for index, label in [
        (0, "a1b1"),
        (97, "e1a1"),
        (103, "e1h1"),
        (322, "e2e4"),
        (1401, "a7a8 (knight)"),
        (1792, "a7a8q"),
        (1793, "a7a8r"),
        (1794, "a7a8b"),
        (1857, "h7h8b"),
    ]:
        print(f"{index:5}  {label:14}  {logits[0, index].item():.0f}")
    print("Castling uses king-to-rook labels; Black's rank flip occurs before mapping.")
    print("Legal-move selection and softmax remain outside this module.")


if __name__ == "__main__":
    main()
