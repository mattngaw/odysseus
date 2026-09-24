"""Inspect value-head shapes, W/D/L probabilities, and search values on CPU."""

import argparse

import torch
from torch.nn import functional as F

from neurodiktyon import SQUARE_COUNT, ValueHead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-hidden", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(0)
    head = ValueHead(args.d_model, args.d_hidden)
    tokens = torch.randn(2, SQUARE_COUNT, args.d_model)
    with torch.inference_mode():
        pooled = tokens.mean(dim=1)
        summary = head.norm(pooled)
        hidden = F.relu(head.in_proj(summary))
        logits = head(tokens)
        torch.testing.assert_close(logits, head.out_proj(hidden))
        probabilities = logits.softmax(dim=-1)
        values = probabilities[:, 0] - probabilities[:, 2]

    print(
        "Synthetic tokens and random weights; these are not trained chess predictions."
    )
    for name, tensor in [
        ("Input square tokens", tokens),
        ("Mean over squares", pooled),
        ("LayerNorm board summary", summary),
        ("Hidden projection + ReLU", hidden),
        ("W/D/L logits (head output)", logits),
        ("W/D/L probabilities (softmax)", probabilities),
        ("Search value: p_win - p_loss", values),
    ]:
        print(f"{name:32} {list(tensor.shape)}")
    print(f"Value-head parameters: {sum(p.numel() for p in head.parameters()):,}")
    print("All outcomes and values are from the side-to-move perspective.")
    for index in range(tokens.shape[0]):
        print(f"Batch {index} logits [W, D, L]: {logits[index].tolist()}")
        win, draw, loss = probabilities[index].tolist()
        print(
            f"  P(win)={win:.6f}, P(draw)={draw:.6f}, P(loss)={loss:.6f}; "
            f"v={values[index].item():+.6f}"
        )


if __name__ == "__main__":
    main()
