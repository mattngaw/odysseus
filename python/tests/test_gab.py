import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from neurodiktyon import GABTemplates, GeometricAttentionBias


def reference_norm(values, norm):
    mean = values.mean(dim=-1, keepdim=True)
    variance = ((values - mean) ** 2).mean(dim=-1, keepdim=True)
    return (values - mean) * torch.rsqrt(variance + norm.eps) * norm.weight + norm.bias


def explicit_gab(gab, tokens):
    """Separate square packing, explicit normalization, and a sum of templates."""
    squares = F.linear(tokens, gab.square_projection.weight, gab.square_projection.bias)
    board = torch.cat([squares[:, square] for square in range(64)], dim=-1)
    summary = reference_norm(
        F.gelu(F.linear(board, gab.board_projection.weight, gab.board_projection.bias)),
        gab.board_norm,
    )
    coefficients = reference_norm(
        F.gelu(F.linear(summary, gab.head_projection.weight, gab.head_projection.bias)),
        gab.head_norm,
    ).reshape(tokens.shape[0], gab.n_heads, gab.templates.d3)
    return sum(
        coefficients[..., r, None, None]
        * gab.templates.projection.weight[:, r].reshape(64, 64)
        for r in range(gab.templates.d3)
    )


def test_matches_explicit_equations_and_gradients():
    torch.manual_seed(0)
    # Distinct dimensions catch accidental use of d_model, d2, or d_head for d3.
    gab = GeometricAttentionBias(12, 3, GABTemplates(5), d1=3, d2=7).double()
    reference = copy.deepcopy(gab)
    tokens = torch.randn(2, 64, 12, dtype=torch.float64, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()

    output = gab(tokens)
    expected = explicit_gab(reference, reference_tokens)
    assert output.shape == (2, 3, 64, 64)
    torch.testing.assert_close(output, expected, rtol=1e-9, atol=1e-10)
    upstream = torch.randn_like(output)
    gradients = torch.autograd.grad(
        (output * upstream).sum(), (tokens, *gab.parameters())
    )
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(), (reference_tokens, *reference.parameters())
    )
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_template_coefficients_are_signed_and_not_probabilities():
    templates = GABTemplates(2)
    with torch.no_grad():
        templates.projection.weight.zero_()
        templates.projection.weight[63, 0] = 2.0  # Query a1, key h8.
        templates.projection.weight[63 * 64, 1] = 3.0  # Query h8, key a1.
    coefficients = torch.tensor([[[2.0, -0.5], [-3.0, 4.0], [0.0, 0.0]]])
    expected = torch.zeros(1, 3, 64, 64)
    expected[0, 0, 0, 63] = 4.0
    expected[0, 0, 63, 0] = -1.5
    expected[0, 1, 0, 63] = -6.0
    expected[0, 1, 63, 0] = 12.0
    torch.testing.assert_close(templates(coefficients), expected, rtol=0, atol=0)


def test_shared_bank_has_one_parameter_and_accumulates_both_layers_gradients():
    torch.manual_seed(0)
    templates = GABTemplates(3).double()
    first = GeometricAttentionBias(8, 2, templates, d1=2, d2=5).double()
    second = GeometricAttentionBias(8, 2, templates, d1=2, d2=5).double()
    layers = nn.ModuleList([first, second])
    assert first.templates is second.templates
    assert first.square_projection is not second.square_projection
    shared_weight = templates.projection.weight
    assert sum(p is shared_weight for p in layers.parameters()) == 1

    tokens = torch.randn(2, 64, 8, dtype=torch.float64)
    first_loss = first(tokens).square().mean()
    second_loss = second(tokens).square().mean()
    first_grad = torch.autograd.grad(first_loss, shared_weight, retain_graph=True)[0]
    second_grad = torch.autograd.grad(second_loss, shared_weight, retain_graph=True)[0]
    combined = torch.autograd.grad(first_loss + second_loss, shared_weight)[0]
    assert first_grad.abs().sum() > 0
    assert second_grad.abs().sum() > 0
    torch.testing.assert_close(
        combined, first_grad + second_grad, rtol=1e-9, atol=1e-10
    )


def test_batch_independence_and_fixed_square_order():
    torch.manual_seed(0)
    gab = GeometricAttentionBias(8, 2, GABTemplates(3), d1=2, d2=5).double()
    tokens = torch.randn(64, 2, 8, dtype=torch.float64).transpose(0, 1)
    output = gab(tokens)
    separate = torch.cat([gab(tokens[i : i + 1]) for i in range(2)], dim=0)
    torch.testing.assert_close(output, separate, rtol=1e-9, atol=1e-10)
    # Moving the same feature vectors among squares changes the global context.
    permutation = torch.randperm(64)
    assert not torch.allclose(
        gab.coefficients(tokens[:, permutation]), gab.coefficients(tokens)
    )


@pytest.mark.parametrize("d3", [0, -1])
def test_rejects_nonpositive_template_count(d3):
    with pytest.raises(ValueError, match="d3 must be positive"):
        GABTemplates(d3)


@pytest.mark.parametrize(
    "overrides", [{"d_model": 0}, {"n_heads": 0}, {"d1": 0}, {"d2": 0}]
)
def test_rejects_nonpositive_generator_dimensions(overrides):
    dimensions = {"d_model": 8, "n_heads": 2, "d1": 2, "d2": 5} | overrides
    with pytest.raises(ValueError, match="must be positive"):
        GeometricAttentionBias(templates=GABTemplates(3), **dimensions)


@pytest.mark.parametrize("shape", [(64, 8), (2, 63, 8), (2, 64, 9)])
def test_rejects_incompatible_token_layout(shape):
    gab = GeometricAttentionBias(8, 2, GABTemplates(3), d1=2, d2=5)
    with pytest.raises(ValueError, match=r"expected \[batch, 64, 8\]"):
        gab(torch.zeros(shape))


@pytest.mark.parametrize("shape", [(2, 3), (2, 2, 4)])
def test_rejects_incompatible_coefficient_layout(shape):
    with pytest.raises(ValueError, match=r"expected \[batch, heads, 3\]"):
        GABTemplates(3)(torch.zeros(shape))
