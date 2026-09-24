import math

import pytest
import torch

from neurodiktyon import (
    POLICY_SIZE,
    ModelOutput,
    TrainingLoss,
    policy_loss,
    training_loss,
    value_loss,
)


def sample():
    logits = torch.zeros(1, POLICY_SIZE, dtype=torch.float64)
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, [0, 159, 322, 1792]] = True
    targets[0, [0, 159, 322]] = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    return logits, targets, mask


def test_known_cross_entropy_and_gradient_including_zero_visit_legal_move():
    logits, targets, mask = sample()
    indices = [0, 159, 322, 1792]
    logits[0, indices] = torch.tensor([2.0, 3.0, 5.0, 10.0]).double().log()
    logits[~mask] = 10_000  # Illegal scores must not enter the denominator.
    logits.requires_grad_()
    loss = policy_loss(logits, targets, mask)
    expected = -(0.6 * math.log(0.1) + 0.3 * math.log(0.15) + 0.1 * math.log(0.25))
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
    torch.testing.assert_close(
        logits.grad[0, indices],
        torch.tensor([-0.5, -0.15, 0.15, 0.5], dtype=torch.float64),
    )


def test_illegal_nan_and_infinities_do_not_change_loss_or_gradients():
    baseline, targets, mask = sample()
    baseline.requires_grad_()
    changed = baseline.detach().clone()
    changed[0, [1, 2, 3]] = torch.tensor([torch.nan, torch.inf, -torch.inf]).double()
    changed.requires_grad_()
    first = policy_loss(baseline, targets, mask)
    second = policy_loss(changed, targets, mask)
    assert first.item() == second.item()
    first.backward()
    second.backward()
    assert torch.equal(baseline.grad, changed.grad)


def test_batch_mean_weights_positions_equally_and_scales_gradients():
    logits, targets, mask = sample()
    # The second position has one legal move: its loss and gradients are zero.
    other_target = torch.zeros_like(targets)
    other_target[0, 1857] = 1
    batched_logits = logits.repeat(2, 1).requires_grad_()
    batched_targets = torch.cat([targets, other_target])
    batched_mask = torch.cat([mask, other_target.bool()])
    loss = policy_loss(batched_logits, batched_targets, batched_mask)
    assert loss.item() == pytest.approx(math.log(4) / 2)
    loss.backward()
    torch.testing.assert_close(
        batched_logits.grad[0, mask[0]], (0.25 - targets[0, mask[0]]) / 2
    )
    assert torch.count_nonzero(batched_logits.grad[1]) == 0


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_widely_separated_logits_have_finite_loss_and_gradients(dtype):
    logits = torch.zeros(1, POLICY_SIZE, dtype=dtype)
    logits[0, :2] = torch.tensor([-10_000.0, 10_000.0], dtype=dtype)
    logits.requires_grad_()
    targets = torch.zeros(1, POLICY_SIZE)
    targets[0, 0] = 1
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, :2] = True
    loss = policy_loss(logits, targets, mask)
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert loss.item() == logits[0, 1].item() - logits[0, 0].item()
    loss.backward()
    torch.testing.assert_close(logits.grad[0, :2], torch.tensor([-1, 1], dtype=dtype))
    assert torch.isfinite(logits.grad).all()


def test_common_legal_logit_shift_preserves_loss():
    logits, targets, mask = sample()
    logits[0, 159] = -3
    shifted = logits.clone()
    shifted[mask] += 10_000
    torch.testing.assert_close(
        policy_loss(logits, targets, mask), policy_loss(shifted, targets, mask)
    )


@pytest.mark.parametrize(
    "bad",
    [
        "empty_legal",
        "zero_target",
        "raw_counts",
        "negative",
        "nan_target",
        "illegal_target",
        "nan_logit",
        "inf_logit",
    ],
)
def test_rejects_invalid_training_examples(bad):
    logits, targets, mask = sample()
    if bad == "empty_legal":
        mask.zero_()
    elif bad == "zero_target":
        targets.zero_()
    elif bad == "raw_counts":
        targets *= 100
    elif bad == "negative":
        targets[0, 0] = -0.6
    elif bad == "nan_target":
        targets[0, 0] = torch.nan
    elif bad == "illegal_target":
        targets[0, 0] -= 0.1
        targets[0, 1] = 0.1
    elif bad == "nan_logit":
        logits[0, 0] = torch.nan
    else:
        logits[0, 0] = torch.inf
    with pytest.raises(ValueError):
        policy_loss(logits, targets, mask)


@pytest.mark.parametrize(
    "shape", [(0, POLICY_SIZE), (POLICY_SIZE,), (2, 64), (2, 1, POLICY_SIZE)]
)
def test_rejects_empty_or_incompatible_layout(shape):
    with pytest.raises(ValueError, match="nonempty logits"):
        policy_loss(
            torch.zeros(shape), torch.zeros(shape), torch.zeros(shape, dtype=torch.bool)
        )


def test_rejects_mismatched_shapes_types_and_devices():
    logits, targets, mask = sample()
    with pytest.raises(ValueError, match="same shape"):
        policy_loss(logits, targets.repeat(2, 1), mask)
    with pytest.raises(TypeError, match="boolean"):
        policy_loss(logits, targets, mask.float())
    with pytest.raises(TypeError, match="floating-point"):
        policy_loss(logits.long(), targets, mask)
    with pytest.raises(ValueError, match="same device"):
        policy_loss(logits, targets.to("meta"), mask)


def test_value_wdl_cross_entropy_and_batch_mean_gradients():
    probabilities = torch.tensor(
        [[0.8, 0.15, 0.05], [0.25, 0.5, 0.25], [0.1, 0.2, 0.7]],
        dtype=torch.float64,
    )
    logits = probabilities.log().requires_grad_()
    targets = torch.eye(3, dtype=torch.float64)  # Win, draw, loss, in that order.
    loss = value_loss(logits, targets)
    assert loss.item() == pytest.approx(
        -(math.log(0.8) + math.log(0.5) + math.log(0.7)) / 3
    )
    loss.backward()
    torch.testing.assert_close(logits.grad, (probabilities - targets) / 3)


def test_value_single_position_and_common_logit_shift():
    logits = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float64).log()
    win = torch.tensor([[1.0, 0.0, 0.0]])
    loss = torch.tensor([[0.0, 0.0, 1.0]])
    assert value_loss(logits, win).item() == pytest.approx(-math.log(0.8))
    assert value_loss(logits, loss).item() == pytest.approx(-math.log(0.1))
    torch.testing.assert_close(
        value_loss(logits, win), value_loss(logits + 10_000, win)
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_value_widely_separated_logits_have_finite_loss_and_gradients(dtype):
    logits = torch.tensor([[-10_000.0, 0.0, 10_000.0]], dtype=dtype, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    loss = value_loss(logits, targets)
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert loss.item() == logits[0, 2].item() - logits[0, 0].item()
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-1, 0, 1]], dtype=dtype))


@pytest.mark.parametrize(
    "target",
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
        [-1.0, 1.0, 1.0],
        [torch.nan, 0.0, 0.0],
        [torch.inf, 0.0, 0.0],
    ],
)
def test_value_rejects_targets_other_than_one_hot_outcomes(target):
    with pytest.raises(ValueError, match="one-hot"):
        value_loss(torch.zeros(1, 3), torch.tensor([target]))


@pytest.mark.parametrize("bad", [torch.nan, torch.inf, -torch.inf])
def test_value_rejects_nonfinite_logits_even_in_zero_target_columns(bad):
    with pytest.raises(ValueError, match="finite"):
        value_loss(torch.tensor([[0.0, bad, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]]))


@pytest.mark.parametrize("shape", [(0, 3), (3,), (2, 4), (2, 1, 3)])
def test_value_rejects_empty_or_incompatible_layout(shape):
    with pytest.raises(ValueError, match="nonempty logits"):
        value_loss(torch.zeros(shape), torch.zeros(shape))


def test_value_rejects_mismatched_shapes_types_and_devices():
    logits = torch.zeros(1, 3)
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="same shape"):
        value_loss(logits, targets.repeat(2, 1))
    with pytest.raises(TypeError, match="floating-point"):
        value_loss(logits.long(), targets)
    with pytest.raises(TypeError, match="floating-point"):
        value_loss(logits, targets.long())
    with pytest.raises(ValueError, match="same device"):
        value_loss(logits, targets.to("meta"))


def test_training_loss_defaults_to_sum_of_unweighted_batch_means():
    logits, targets, mask = sample()
    output = ModelOutput(
        logits.repeat(2, 1).requires_grad_(),
        torch.tensor([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]], dtype=torch.float64)
        .log()
        .requires_grad_(),
    )
    outcomes = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    losses = training_loss(output, targets.repeat(2, 1), outcomes, mask.repeat(2, 1))
    assert isinstance(losses, TrainingLoss)
    assert all(
        component.shape == () and component.requires_grad for component in losses
    )
    expected_value = -(math.log(0.5) + math.log(0.8)) / 2
    assert losses.policy.item() == pytest.approx(math.log(4))
    assert losses.value.item() == pytest.approx(expected_value)
    assert losses.total.item() == pytest.approx(math.log(4) + expected_value)


@pytest.mark.parametrize("weight", [0.0, 0.5, 2.0])
def test_training_weight_preserves_components_and_scales_only_value_gradients(weight):
    logits, targets, mask = sample()
    probabilities = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    output = ModelOutput(logits.requires_grad_(), probabilities.log().requires_grad_())
    outcomes = torch.tensor([[0.0, 0.0, 1.0]])
    losses = training_loss(output, targets, outcomes, mask, value_weight=weight)
    assert losses.policy.item() == pytest.approx(math.log(4))
    assert losses.value.item() == pytest.approx(math.log(2))
    assert losses.total.item() == pytest.approx(math.log(4) + weight * math.log(2))
    losses.total.backward()
    expected_policy = torch.zeros_like(logits)
    expected_policy[mask] = 0.25 - targets[mask]
    torch.testing.assert_close(output.policy_logits.grad, expected_policy)
    torch.testing.assert_close(
        output.value_logits.grad, weight * (probabilities - outcomes)
    )


@pytest.mark.parametrize("weight", [-1.0, math.nan, math.inf, -math.inf])
def test_training_loss_rejects_invalid_value_weight(weight):
    logits, targets, mask = sample()
    output = ModelOutput(logits, torch.zeros(1, 3))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        training_loss(
            output, targets, torch.tensor([[1.0, 0.0, 0.0]]), mask, value_weight=weight
        )


def test_training_loss_rejects_learnable_weight_and_mismatched_head_batches_or_devices():
    logits, targets, mask = sample()
    values = torch.zeros(1, 3)
    outcomes = torch.tensor([[1.0, 0.0, 0.0]])
    with pytest.raises(TypeError, match="Python number"):
        training_loss(
            ModelOutput(logits, values),
            targets,
            outcomes,
            mask,
            value_weight=torch.tensor(1.0, requires_grad=True),
        )
    with pytest.raises(ValueError, match="same batch size"):
        training_loss(
            ModelOutput(logits, values.repeat(2, 1)),
            targets,
            outcomes.repeat(2, 1),
            mask,
        )
    with pytest.raises(ValueError, match="same device"):
        training_loss(ModelOutput(logits, values.to("meta")), targets, outcomes, mask)


def test_training_loss_still_validates_value_targets_at_zero_weight():
    logits, targets, mask = sample()
    output = ModelOutput(logits, torch.zeros(1, 3))
    with pytest.raises(ValueError, match="one-hot"):
        training_loss(output, targets, torch.zeros(1, 3), mask, value_weight=0.0)
