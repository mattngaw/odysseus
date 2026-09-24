"""Inspect legal-only policy loss and gradients for an illustrative start position."""

import torch

from neurodiktyon import POLICY_SIZE, policy_loss
from neurodiktyon._policy_map_data import BASE_INDICES


def index(move: str) -> int:
    def square(text: str) -> int:
        return (int(text[1]) - 1) * 8 + ord(text[0]) - ord("a")

    return BASE_INDICES.index(square(move[:2]) * 64 + square(move[2:]))


def main() -> None:
    moves = [f"{file}2{file}{rank}" for file in "abcdefgh" for rank in "34"]
    moves += ["b1a3", "b1c3", "g1f3", "g1h3"]
    legal = torch.zeros(1, POLICY_SIZE, dtype=torch.bool)
    legal[0, [index(move) for move in moves]] = True
    visits = torch.zeros(1, POLICY_SIZE)
    visits[0, [index(move) for move in ["e2e4", "d2d4", "g1f3"]]] = torch.tensor(
        [60.0, 30.0, 10.0]
    )
    targets = visits / visits.sum(dim=1, keepdim=True)
    logits = torch.full((1, POLICY_SIZE), 50.0)
    logits[legal] = 0
    logits.requires_grad_()
    loss = policy_loss(logits, targets, legal)
    loss.backward()
    probabilities = logits.detach().masked_fill(~legal, -torch.inf).softmax(dim=1)

    print("Synthetic logits and visits; 20 legal moves in the initial position.")
    print("Legal logits are 0; illegal logits are 50 and excluded from softmax.")
    print(f"Batch mean cross-entropy: {loss.item():.6f} nats (ln 20)")
    print("Move    Legal   Visits   Target   Predicted   dLoss/dLogit")
    for move in ["e2e4", "d2d4", "g1f3", "b1a3", "e2e5"]:
        slot = index(move)
        print(
            f"{move:6}  {str(legal[0, slot].item()):5}  {visits[0, slot]:6.0f}"
            f"   {targets[0, slot]:.2f}     {probabilities[0, slot]:.2f}"
            f"        {logits.grad[0, slot]:+.2f}"
        )
    print("Gradient descent raises negative-gradient logits and lowers positive ones.")
    print(
        f"Nonzero illegal-logit gradients: {torch.count_nonzero(logits.grad[~legal]).item()}"
    )


if __name__ == "__main__":
    main()
