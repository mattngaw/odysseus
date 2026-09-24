import math

import pytest
import torch

from neurodiktyon import PolicyPairScorer


def known_scorer_and_tokens():
    scorer = PolicyPairScorer(3, 2).double()
    with torch.no_grad():
        scorer.from_proj.weight.copy_(torch.tensor([[1, 0, 0], [0, 1, 0]]))
        scorer.from_proj.bias.copy_(torch.tensor([1, -2]))
        scorer.to_proj.weight.copy_(torch.tensor([[0, 0, 1], [1, 0, 0]]))
        scorer.to_proj.bias.copy_(torch.tensor([-1, 3]))
    tokens = torch.zeros(2, 64, 3, dtype=torch.float64)
    tokens[0, 12] = torch.tensor([2, 3, 5])
    tokens[0, 28] = torch.tensor([-1, 4, 2])
    return scorer, tokens


def test_known_directed_scores_use_both_biases_and_policy_width_scaling():
    scorer, tokens = known_scorer_and_tokens()
    logits = scorer(tokens)
    assert logits.shape == (2, 64, 64)
    # e2: Q=(3,1), K=(4,5); e4: Q=(0,2), K=(1,2).
    actual = torch.stack(
        [logits[0, 12, 28], logits[0, 28, 12], logits[0, 12, 12], logits[1, 0, 0]]
    )
    expected = torch.tensor([5, 10, 17, -7], dtype=torch.float64) / math.sqrt(2)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_one_pair_has_exact_input_and_projection_gradients():
    scorer, tokens = known_scorer_and_tokens()
    tokens.requires_grad_()
    scorer(tokens)[0, 12, 28].backward()
    scale = 1 / math.sqrt(2)

    # d(Q_s dot K_t)/dx_s = W_from.T K_t; d/dx_t = W_to.T Q_s.
    expected_input = torch.zeros_like(tokens)
    expected_input[0, 12] = torch.tensor([1, 2, 0], dtype=torch.float64) * scale
    expected_input[0, 28] = torch.tensor([1, 0, 3], dtype=torch.float64) * scale
    torch.testing.assert_close(tokens.grad, expected_input, rtol=1e-9, atol=1e-10)

    expected_gradients = [
        (scorer.from_proj.weight, [[2, 3, 5], [4, 6, 10]]),
        (scorer.from_proj.bias, [1, 2]),
        (scorer.to_proj.weight, [[-3, 12, 6], [-1, 4, 2]]),
        (scorer.to_proj.bias, [3, 1]),
    ]
    for parameter, expected in expected_gradients:
        torch.testing.assert_close(
            parameter.grad,
            torch.tensor(expected, dtype=torch.float64) * scale,
            rtol=1e-9,
            atol=1e-10,
        )


def test_batches_are_independent_and_square_permutations_reorder_both_axes():
    torch.manual_seed(0)
    scorer = PolicyPairScorer(7, 5).double()
    tokens = torch.randn(64, 2, 7, dtype=torch.float64).transpose(0, 1)
    assert not tokens.is_contiguous()
    logits = scorer(tokens)
    separate = torch.cat([scorer(tokens[i : i + 1]) for i in range(2)], dim=0)
    torch.testing.assert_close(logits, separate, rtol=1e-9, atol=1e-10)

    permutation = torch.randperm(64)
    torch.testing.assert_close(
        scorer(tokens[:, permutation]),
        logits[:, permutation][:, :, permutation],
        rtol=1e-9,
        atol=1e-10,
    )


@pytest.mark.parametrize("d_model,d_policy", [(0, 2), (-1, 2), (3, 0), (3, -1)])
def test_rejects_nonpositive_dimensions(d_model, d_policy):
    with pytest.raises(ValueError, match="d_model and d_policy must be positive"):
        PolicyPairScorer(d_model, d_policy)


@pytest.mark.parametrize("shape", [(64, 3), (2, 63, 3), (2, 64, 2)])
def test_rejects_incompatible_token_layout(shape):
    with pytest.raises(ValueError, match=r"expected \[batch, 64, 3\]"):
        PolicyPairScorer(3, 2)(torch.zeros(shape))
