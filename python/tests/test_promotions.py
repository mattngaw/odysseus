import math

import pytest
import torch

from neurodiktyon import PolicyPairScorer, PromotionScorer


def known_promotions():
    scorer = PromotionScorer(2).double()
    with torch.no_grad():
        scorer.projection.weight.copy_(torch.tensor([[1, 2], [-1, 0], [0, 3], [2, -1]]))
    return scorer


def test_known_offsets_rank_slices_piece_order_and_unchanged_base_scores():
    scorer = known_promotions()
    destinations = torch.zeros(2, 64, 2, dtype=torch.float64)
    destinations[0, 60] = torch.tensor([3, -2])  # e8 -> offsets [7, 5, 2].
    destinations[0, 56] = torch.tensor([-1, 4])  # a8 -> offsets [1, -5, 6].
    destinations[:, :56] = 100  # Other ranks must not contribute offsets.
    pairs = torch.zeros(2, 64, 64, dtype=torch.float64)
    pairs[0, 51, 60] = 1.5  # d7 -> e8.
    pairs[0, 52, 60] = -2  # e7 -> e8.
    pairs[0, 48, 56] = 10  # a7 -> a8.
    pairs[1, 50, 58] = 4  # c7 -> c8, in a separate batch item.
    original_pairs = pairs.clone()
    original_destinations = destinations.clone()

    expected = torch.zeros(2, 8, 8, 3, dtype=torch.float64)
    expected[0, :, 4] = torch.tensor([7, 5, 2])
    expected[0, 3, 4] += 1.5
    expected[0, 4, 4] -= 2
    expected[0, :, 0] = torch.tensor([1, -5, 6])
    expected[0, 0, 0] += 10
    expected[1, 2, 2] = 4
    torch.testing.assert_close(scorer(pairs, destinations), expected, rtol=0, atol=0)
    assert torch.equal(pairs, original_pairs)
    assert torch.equal(destinations, original_destinations)


def test_exact_gradients_include_the_common_adjustment_for_all_three_pieces():
    scorer = known_promotions()
    destinations = torch.zeros(1, 64, 2, dtype=torch.float64)
    destinations[0, 60] = torch.tensor([3, -2])
    destinations.requires_grad_()
    pairs = torch.zeros(1, 64, 64, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([2, -1, 3], dtype=torch.float64)
    (scorer(pairs, destinations)[0, 3, 4] * weights).sum().backward()

    expected_pairs = torch.zeros_like(pairs)
    expected_pairs[0, 51, 60] = 4  # Sum of the three upstream gradients.
    expected_destinations = torch.zeros_like(destinations)
    expected_destinations[0, 60] = torch.tensor([11, 9])
    expected_weights = torch.tensor(
        [[6, -4], [-3, 2], [9, -6], [12, -8]], dtype=torch.float64
    )
    torch.testing.assert_close(pairs.grad, expected_pairs, rtol=0, atol=0)
    torch.testing.assert_close(destinations.grad, expected_destinations, rtol=0, atol=0)
    torch.testing.assert_close(
        scorer.projection.weight.grad, expected_weights, rtol=0, atol=0
    )


def test_destination_projection_is_reused_and_receives_both_gradient_paths():
    pairs = PolicyPairScorer(2, 2).double()
    promotions = known_promotions()
    with torch.no_grad():
        for projection in (pairs.from_proj, pairs.to_proj):
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
    tokens = torch.zeros(1, 64, 2, dtype=torch.float64)
    tokens[0, 51] = torch.tensor([1, 4])
    tokens[0, 60] = torch.tensor([3, -2])
    tokens.requires_grad_()

    # Observe the actual projection output to check reuse without recomputation.
    projected = []
    handle = pairs.to_proj.register_forward_hook(
        lambda module, inputs, output: projected.append(output)
    )
    try:
        pair_logits, destinations = pairs.forward_with_destinations(tokens)
        result = promotions(pair_logits, destinations)
    finally:
        handle.remove()
    assert len(projected) == 1
    assert destinations is projected[0]
    torch.testing.assert_close(
        result[0, 3, 4, 0], torch.tensor(7 - 5 / math.sqrt(2), dtype=torch.float64)
    )
    result[0, 3, 4, 0].backward()  # Queen promotion d7 -> e8.

    expected_input = torch.zeros_like(tokens)
    expected_input[0, 51] = torch.tensor([3, -2], dtype=torch.float64) / math.sqrt(2)
    expected_input[0, 60] = torch.tensor(
        [1 / math.sqrt(2) + 3, 4 / math.sqrt(2) + 1], dtype=torch.float64
    )
    torch.testing.assert_close(tokens.grad, expected_input, rtol=1e-9, atol=1e-10)
    expected_destination_weight = torch.outer(
        expected_input[0, 60], tokens.detach()[0, 60]
    )
    torch.testing.assert_close(
        pairs.to_proj.weight.grad, expected_destination_weight, rtol=1e-9, atol=1e-10
    )


def test_zero_destination_vectors_give_zero_offsets_without_a_bias():
    scorer = PromotionScorer(5).double()
    destinations = torch.zeros(2, 64, 5, dtype=torch.float64)
    assert scorer.projection.bias is None
    assert torch.equal(
        scorer.offsets(destinations), torch.zeros(2, 8, 3, dtype=torch.float64)
    )


def test_batches_are_independent_with_noncontiguous_inputs():
    torch.manual_seed(0)
    scorer = PromotionScorer(5).double()
    destinations = torch.randn(64, 2, 5, dtype=torch.float64).transpose(0, 1)
    pairs = torch.randn(64, 2, 64, dtype=torch.float64).transpose(0, 1)
    assert not destinations.is_contiguous() and not pairs.is_contiguous()
    separate = torch.cat(
        [scorer(pairs[i : i + 1], destinations[i : i + 1]) for i in range(2)], dim=0
    )
    torch.testing.assert_close(scorer(pairs, destinations), separate)


@pytest.mark.parametrize("d_policy", [0, -1])
def test_rejects_nonpositive_policy_width(d_policy):
    with pytest.raises(ValueError, match="d_policy must be positive"):
        PromotionScorer(d_policy)


@pytest.mark.parametrize("shape", [(64, 5), (2, 63, 5), (2, 64, 4)])
def test_rejects_incompatible_destination_layout(shape):
    with pytest.raises(ValueError, match=r"expected destinations \[batch, 64, 5\]"):
        PromotionScorer(5).offsets(torch.zeros(shape))


@pytest.mark.parametrize("shape", [(64, 64), (2, 64, 63), (1, 64, 64)])
def test_pair_logits_require_matching_batch_and_both_square_axes(shape):
    with pytest.raises(ValueError, match="expected pair_logits shape"):
        PromotionScorer(5)(torch.zeros(shape), torch.zeros(2, 64, 5))
