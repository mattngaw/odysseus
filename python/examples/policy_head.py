"""Inspect the assembled policy head's output and parameter counts on CPU."""

import argparse

import torch

from neurodiktyon import SQUARE_COUNT, PolicyHead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-policy", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(0)
    head = PolicyHead(args.d_model, args.d_policy)
    tokens = torch.randn(2, SQUARE_COUNT, args.d_model)
    with torch.inference_mode():
        logits = head(tokens)

    print("Synthetic tokens and random weights; no trained chess predictions.")
    print(f"Input square tokens: {list(tokens.shape)}")
    print(f"Raw logits in Pyxis vocabulary order: {list(logits.shape)}")
    for name, component in [
        ("Pair scorer", head.pairs),
        ("Promotion scorer", head.promotions),
        ("Vocabulary map", head.mapping),
        ("Complete policy head", head),
    ]:
        print(
            f"{name:22} {sum(p.numel() for p in component.parameters()):>8,} parameters"
        )
    print("Index  Relative entry       Batch 0 logit")
    for index, label in [
        (97, "e1a1 (castling)"),
        (103, "e1h1 (castling)"),
        (322, "e2e4"),
        (1401, "a7a8 (knight)"),
        (1792, "a7a8q"),
        (1793, "a7a8r"),
        (1794, "a7a8b"),
    ]:
        print(f"{index:5}  {label:20} {logits[0, index].item():+.6f}")
    print("Legal-move selection and softmax remain outside this head.")


if __name__ == "__main__":
    main()
