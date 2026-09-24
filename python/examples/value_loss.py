"""Inspect W/D/L targets, cross-entropy, and gradients for completed-game examples."""

import torch

from neurodiktyon import value_loss


def main() -> None:
    labels = [
        "White won; White to move",
        "White won; Black to move",
        "Drawn game; either side",
    ]
    # The first two positions come from the same completed game. Each target
    # describes the result for the player to move in that recorded position.
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    logits = torch.tensor([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.2, 0.6, 0.2]]).log()
    logits.requires_grad_()
    loss = value_loss(logits, targets)
    loss.backward()
    probabilities = logits.detach().softmax(dim=1)

    print("Synthetic predictions; all vectors use [win, draw, loss] order.")
    print("Recorded position          Target W/D/L   Predicted W/D/L    Loss (nats)")
    for i, label in enumerate(labels):
        target = "/".join(f"{x:.0f}" for x in targets[i].tolist())
        predicted = "/".join(f"{x:.2f}" for x in probabilities[i].tolist())
        individual = value_loss(logits.detach()[i : i + 1], targets[i : i + 1])
        print(f"{label:26} {target:14} {predicted:18} {individual.item():.6f}")
    print(f"Batch mean loss: {loss.item():.6f} nats")
    print("Gradients of the batch mean with respect to each row's W/D/L logits:")
    for label, gradient in zip(labels, logits.grad, strict=True):
        print(f"  {label:26} " + ", ".join(f"{x:+.6f}" for x in gradient.tolist()))
    print("Gradient descent raises the observed outcome's logit and lowers the others.")


if __name__ == "__main__":
    main()
