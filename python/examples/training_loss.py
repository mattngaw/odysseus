"""Inspect the combined objective and the effect of its value weight."""

import torch

from neurodiktyon import POLICY_SIZE, ModelOutput, training_loss


def main() -> None:
    policy_targets = torch.zeros(1, POLICY_SIZE)
    policy_targets[0, :3] = torch.tensor([0.6, 0.3, 0.1])
    legal_mask = torch.zeros_like(policy_targets, dtype=torch.bool)
    legal_mask[0, :3] = True
    value_targets = torch.tensor([[1.0, 0.0, 0.0]])

    print("Synthetic legal slots, logits, and targets; these are not recorded games.")
    print("Policy: uniform predictions over 3 legal slots; target [0.6, 0.3, 0.1].")
    print("Value: predicted W/D/L [0.2, 0.3, 0.5]; target [1, 0, 0] (win).")
    print(
        "Total = policy + value_weight * value; components are unweighted batch means."
    )
    print("Gradients below are with respect to output logits.")
    for weight in [0.0, 0.5, 1.0, 2.0]:
        output = ModelOutput(
            torch.zeros_like(policy_targets, requires_grad=True),
            torch.tensor([[0.2, 0.3, 0.5]]).log().requires_grad_(),
        )
        losses = training_loss(
            output, policy_targets, value_targets, legal_mask, value_weight=weight
        )
        losses.total.backward()
        print(
            f"\nvalue_weight={weight:.1f}  policy={losses.policy.item():.6f}"
            f"  value={losses.value.item():.6f}  total={losses.total.item():.6f}"
        )
        policy_grad = output.policy_logits.grad[legal_mask].tolist()
        value_grad = output.value_logits.grad[0].tolist()
        print(
            "  Legal policy gradients: " + ", ".join(f"{x:+.6f}" for x in policy_grad)
        )
        print("  W/D/L gradients:        " + ", ".join(f"{x:+.6f}" for x in value_grad))


if __name__ == "__main__":
    main()
