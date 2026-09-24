import math

import pytest
import torch

from neurodiktyon import FeedForward


def test_known_nonlinear_mapping_with_both_biases():
    ffn = FeedForward(3, 2).double()
    with torch.no_grad():
        ffn.in_proj.weight.copy_(torch.tensor([[1, 0, 0], [0, 1, 0]]))
        ffn.in_proj.bias.copy_(torch.tensor([0.5, -0.5]))
        ffn.out_proj.weight.copy_(torch.tensor([[1, 0], [0, 1], [1, -1]]))
        ffn.out_proj.bias.copy_(torch.tensor([1, 2, 3]))

    tokens = torch.zeros(1, 64, 3, dtype=torch.float64)
    tokens[0, 0] = torch.tensor([1.0, -1.0, 4.0])

    # Exact GELU(x) = x * Phi(x), evaluated with a scalar normal CDF.
    positive = 1.5 * (1 + math.erf(1.5 / math.sqrt(2))) / 2
    negative = -1.5 * (1 + math.erf(-1.5 / math.sqrt(2))) / 2
    expected = torch.tensor(
        [positive + 1, negative + 2, positive - negative + 3], dtype=torch.float64
    )
    torch.testing.assert_close(ffn(tokens)[0, 0], expected)


@pytest.mark.parametrize("d_ff", [5, 12, 24])
def test_shared_square_mapping_and_local_gradients(d_ff):
    torch.manual_seed(0)
    ffn = FeedForward(12, d_ff).double()
    tokens = torch.randn(2, 64, 12, dtype=torch.float64, requires_grad=True)
    output = ffn(tokens)
    assert output.shape == tokens.shape

    permutation = torch.randperm(64)
    torch.testing.assert_close(ffn(tokens[:, permutation]), output[:, permutation])
    changed = tokens.detach().clone()
    changed[0, 8, 0] += 1.0
    affected = (ffn(changed) != output).any(dim=-1).nonzero().tolist()
    assert affected == [[0, 8]]

    output[0, 8].square().sum().backward()
    assert tokens.grad is not None
    assert (tokens.grad != 0).any(dim=-1).nonzero().tolist() == [[0, 8]]
    for parameter in ffn.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("d_model, d_ff", [(0, 8), (-1, 8), (8, 0), (8, -1)])
def test_rejects_nonpositive_dimensions(d_model, d_ff):
    with pytest.raises(ValueError, match="d_model and d_ff must be positive"):
        FeedForward(d_model, d_ff)


@pytest.mark.parametrize("shape", [(64, 12), (2, 63, 12), (2, 64, 11)])
def test_rejects_incompatible_token_layout(shape):
    with pytest.raises(ValueError, match=r"expected \[batch, 64, 12\]"):
        FeedForward(12, 5)(torch.zeros(shape))
