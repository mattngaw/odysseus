import math

import pytest
import torch

from neurodiktyon import TransformerTrunk, ValueHead, value_loss


def test_known_mean_normalization_relu_and_raw_logits():
    head = ValueHead(4, 3).double()
    with torch.no_grad():
        head.in_proj.weight.copy_(torch.eye(4)[[0, 1, 3]])
        head.in_proj.bias.copy_(torch.tensor([0.25, -0.5, 0.75]))
        head.out_proj.weight.copy_(torch.eye(3))
        head.out_proj.bias.copy_(torch.tensor([-2.0, 3.0, 0.5]))
    tokens = torch.empty(1, 64, 4, dtype=torch.float64)
    tokens[:, :32] = 2.0
    tokens[:, 32:] = torch.tensor([-4.0, -2.0, 0.0, 2.0])

    # Pooled features [-1, 0, 1, 2] have mean 0.5 and variance 1.25.
    # ReLU zeros the first two hidden activations. Outputs remain raw logits.
    expected = torch.tensor(
        [[-2.0, 3.0, 1.5 / math.sqrt(1.25 + 1e-5) + 1.25]], dtype=torch.float64
    )
    torch.testing.assert_close(head(tokens), expected, rtol=1e-9, atol=1e-10)


def test_redistributing_features_between_squares_preserves_the_prediction():
    torch.manual_seed(0)
    head = ValueHead(12, 7).double()
    tokens = torch.randn(2, 64, 12, dtype=torch.float64)
    changed = tokens.clone()
    delta = torch.randn(12, dtype=torch.float64) * 5
    changed[0, 0] += delta
    changed[0, 1] -= delta

    # The board summary is unchanged. Nonlinear processing must follow pooling.
    torch.testing.assert_close(head(tokens), head(changed), rtol=1e-9, atol=1e-10)


def test_batches_are_independent_with_noncontiguous_tokens():
    torch.manual_seed(0)
    head = ValueHead(12, 7).double()
    tokens = torch.randn(64, 2, 12, dtype=torch.float64).transpose(0, 1)
    assert not tokens.is_contiguous()
    separate = torch.cat([head(tokens[i : i + 1]) for i in range(2)], dim=0)
    torch.testing.assert_close(head(tokens), separate, rtol=1e-9, atol=1e-10)


def test_classification_loss_reaches_the_head_and_entire_trunk():
    torch.manual_seed(0)
    trunk = TransformerTrunk(
        n_blocks=2, d_model=12, n_heads=3, d_ff=19, gab_d1=3, gab_d2=7, gab_d3=5
    ).double()
    head = ValueHead(12, 7).double()
    features = torch.randn(3, 64, 110, dtype=torch.float64, requires_grad=True)
    tokens = trunk(features)
    tokens.retain_grad()
    logits = head(tokens)
    assert logits.shape == (3, 3)
    targets = torch.eye(3, dtype=torch.float64)  # Synthetic win, draw, loss labels.
    value_loss(logits, targets).backward()

    named_tensors = [("input", features)]
    named_tensors += [(f"trunk.{name}", p) for name, p in trunk.named_parameters()]
    named_tensors += [(f"head.{name}", p) for name, p in head.named_parameters()]
    for name, tensor in named_tensors:
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum() > 0, name
    # Mean pooling distributes a board summary's gradient equally to all squares.
    assert tokens.grad is not None
    torch.testing.assert_close(
        tokens.grad, tokens.grad[:, :1].expand_as(tokens.grad), rtol=0, atol=0
    )


@pytest.mark.parametrize("d_model,d_hidden", [(0, 7), (-1, 7), (12, 0), (12, -1)])
def test_rejects_nonpositive_dimensions(d_model, d_hidden):
    with pytest.raises(ValueError, match="d_model and d_hidden must be positive"):
        ValueHead(d_model, d_hidden)


@pytest.mark.parametrize("shape", [(64, 12), (2, 63, 12), (2, 64, 11)])
def test_rejects_incompatible_token_layout(shape):
    with pytest.raises(ValueError, match=r"expected \[batch, 64, 12\]"):
        ValueHead(12, 7)(torch.zeros(shape))
