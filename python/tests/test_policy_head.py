import math

import torch

from neurodiktyon import PolicyHead, TransformerTrunk, policy_loss


def test_known_scores_reach_ordinary_castling_and_all_four_promotion_slots():
    head = PolicyHead(2, 2).double()
    with torch.no_grad():
        for projection in (head.pairs.from_proj, head.pairs.to_proj):
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
        head.promotions.projection.weight.copy_(
            torch.tensor([[1.0, 2.0], [-1.0, 0.0], [0.0, 3.0], [2.0, -1.0]])
        )
    tokens = torch.zeros(1, 64, 2, dtype=torch.float64)
    tokens[0, 0] = torch.tensor([1.0, -3.0])  # a1
    tokens[0, 4] = torch.tensor([2.0, 1.0])  # e1
    tokens[0, 7] = torch.tensor([-2.0, 1.0])  # h1
    tokens[0, 12] = torch.tensor([2.0, 3.0])  # e2
    tokens[0, 28] = torch.tensor([-1.0, 4.0])  # e4
    tokens[0, 48] = torch.tensor([1.0, 4.0])  # a7
    tokens[0, 56] = torch.tensor([3.0, -2.0])  # a8

    logits = head(tokens)
    assert logits.shape == (1, 1858)
    # Castling is king-to-rook. Knight promotion keeps the a7a8 base score.
    # K_a8 gives raw Q/R/B/common [-1, -3, -6, 8], hence offsets [7, 5, 2].
    base = -5 / math.sqrt(2)
    expected = torch.tensor(
        [
            -1 / math.sqrt(2),
            -3 / math.sqrt(2),
            10 / math.sqrt(2),
            base,
            base + 7,
            base + 5,
            base + 2,
        ],
        dtype=torch.float64,
    )
    indices = [97, 103, 322, 1401, 1792, 1793, 1794]
    torch.testing.assert_close(logits[0, indices], expected, rtol=1e-9, atol=1e-10)


def test_policy_loss_reaches_both_scoring_branches_and_entire_trunk():
    torch.manual_seed(0)
    trunk = TransformerTrunk(
        n_blocks=2, d_model=12, n_heads=3, d_ff=19, gab_d1=3, gab_d2=7, gab_d3=5
    ).double()
    head = PolicyHead(12, 7).double()
    features = torch.randn(3, 64, 110, dtype=torch.float64, requires_grad=True)
    logits = head(trunk(features))
    assert logits.shape == (3, 1858)
    # Synthetic move distributions exercise ordinary moves and promotions.
    targets = torch.zeros_like(logits)
    targets[0, [159, 322]] = torch.tensor([0.4, 0.6], dtype=torch.float64)
    targets[1, [1401, 1792]] = torch.tensor([0.2, 0.8], dtype=torch.float64)
    targets[2, [1793, 1794]] = torch.tensor([0.7, 0.3], dtype=torch.float64)
    legal = torch.zeros_like(targets, dtype=torch.bool)
    legal[0, [0, 159, 322]] = True
    legal[1:, [1401, 1792, 1793, 1794]] = True
    policy_loss(logits, targets, legal).backward()

    named_tensors = [("input", features)]
    named_tensors += [(f"trunk.{name}", p) for name, p in trunk.named_parameters()]
    named_tensors += [(f"head.{name}", p) for name, p in head.named_parameters()]
    for name, tensor in named_tensors:
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum() > 0, name


def test_complete_head_computes_destination_projection_once():
    head = PolicyHead(12, 7)
    calls = []
    hook = head.pairs.to_proj.register_forward_hook(
        lambda module, inputs, output: calls.append(output)
    )
    try:
        head(torch.randn(2, 64, 12))
    finally:
        hook.remove()
    assert len(calls) == 1
