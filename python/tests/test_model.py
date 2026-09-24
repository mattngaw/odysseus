import pytest
import torch

from neurodiktyon import ChessModel, ModelOutput, training_loss


def small_model():
    return ChessModel(
        n_blocks=2,
        d_model=12,
        n_heads=3,
        d_ff=19,
        gab_d1=3,
        gab_d2=7,
        gab_d3=5,
        d_policy=7,
        d_value_hidden=9,
    ).double()


def test_both_heads_receive_the_same_single_trunk_result_and_return_raw_logits():
    model = small_model()
    # Known biases distinguish raw logits from probabilities or a scalar value.
    with torch.no_grad():
        model.policy_head.pairs.from_proj.weight.zero_()
        model.policy_head.pairs.from_proj.bias.zero_()
        model.policy_head.promotions.projection.weight.zero_()
        model.value_head.out_proj.weight.zero_()
        model.value_head.out_proj.bias.copy_(torch.tensor([-2.0, 3.0, 0.5]))

    trunk_results, policy_inputs, value_inputs = [], [], []
    hooks = [
        model.trunk.register_forward_hook(
            lambda module, inputs, output: trunk_results.append(output)
        ),
        model.policy_head.register_forward_pre_hook(
            lambda module, inputs: policy_inputs.append(inputs[0])
        ),
        model.value_head.register_forward_pre_hook(
            lambda module, inputs: value_inputs.append(inputs[0])
        ),
    ]
    try:
        output = model(torch.randn(2, 64, 110, dtype=torch.float64))
    finally:
        for hook in hooks:
            hook.remove()

    assert len(trunk_results) == len(policy_inputs) == len(value_inputs) == 1
    assert policy_inputs[0] is value_inputs[0] is trunk_results[0]
    assert trunk_results[0].shape == (2, 64, 12)
    assert isinstance(output, ModelOutput)
    torch.testing.assert_close(
        output.policy_logits, torch.zeros(2, 1858, dtype=torch.float64)
    )
    torch.testing.assert_close(
        output.value_logits,
        torch.tensor([[-2.0, 3.0, 0.5]], dtype=torch.float64).expand(2, 3),
    )


@pytest.mark.parametrize("value_weight", [0.0, 0.5, 1.0, 2.0])
def test_joint_loss_adds_weighted_gradients_in_the_shared_trunk(value_weight):
    torch.manual_seed(0)
    model = small_model()
    features = torch.randn(3, 64, 110, dtype=torch.float64, requires_grad=True)
    output = model(features)
    # Synthetic targets exercise both heads and their paths through the trunk.
    policy_targets = torch.zeros_like(output.policy_logits)
    policy_targets[0, [159, 322]] = torch.tensor([0.4, 0.6], dtype=torch.float64)
    policy_targets[1, [1401, 1792]] = torch.tensor([0.2, 0.8], dtype=torch.float64)
    policy_targets[2, [1793, 1794]] = torch.tensor([0.7, 0.3], dtype=torch.float64)
    legal = torch.zeros_like(policy_targets, dtype=torch.bool)
    legal[0, [0, 159, 322]] = True
    legal[1:, [1401, 1792, 1793, 1794]] = True
    value_targets = torch.eye(3, dtype=torch.float64)  # Win, draw, loss.
    losses = training_loss(
        output, policy_targets, value_targets, legal, value_weight=value_weight
    )

    named_tensors = [("input", features), *model.named_parameters()]
    tensors = [tensor for _, tensor in named_tensors]
    policy_gradients = torch.autograd.grad(
        losses.policy, tensors, allow_unused=True, retain_graph=True
    )
    value_gradients = torch.autograd.grad(
        losses.value, tensors, allow_unused=True, retain_graph=True
    )
    losses.total.backward()

    for (name, tensor), policy_grad, value_grad in zip(
        named_tensors, policy_gradients, value_gradients, strict=True
    ):
        assert (policy_grad is None) == name.startswith("value_head."), name
        assert (value_grad is None) == name.startswith("policy_head."), name
        for gradient in (policy_grad, value_grad):
            if gradient is not None:
                assert torch.isfinite(gradient).all(), name
                assert gradient.abs().sum() > 0, name
        expected = (
            policy_grad if policy_grad is not None else torch.zeros_like(tensor)
        ) + value_weight * (
            value_grad if value_grad is not None else torch.zeros_like(tensor)
        )
        assert tensor.grad is not None, name
        torch.testing.assert_close(tensor.grad, expected, rtol=1e-9, atol=1e-10)
