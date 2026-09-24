import copy
import math

import pytest
import torch
from torch.nn import functional as F

from neurodiktyon import GABTemplates, GeometricAttentionBias, MultiHeadSelfAttention


def explicit_attention(module, tokens, attention_bias=None):
    """Reference: separate Q/K/V projections and a loop over individual heads."""
    weights = module.qkv.weight.chunk(3, dim=0)
    biases = module.qkv.bias.chunk(3, dim=0)
    q, k, v = (
        F.linear(tokens, weight, bias)
        for weight, bias in zip(weights, biases, strict=True)
    )
    outputs = []
    for head in range(module.n_heads):
        span = slice(head * module.d_head, (head + 1) * module.d_head)
        scores = (q[..., span] @ k[..., span].transpose(-2, -1)) / math.sqrt(
            module.d_head
        )
        if attention_bias is not None:
            scores = scores + attention_bias[:, head]
        outputs.append(scores.softmax(dim=-1) @ v[..., span])
    return F.linear(
        torch.cat(outputs, dim=-1), module.out_proj.weight, module.out_proj.bias
    )


@pytest.mark.parametrize("d_model,n_heads", [(8, 1), (12, 3), (32, 4)])
def test_matches_explicit_equations_and_gradients(d_model, n_heads):
    torch.manual_seed(0)
    attention = MultiHeadSelfAttention(d_model, n_heads).double()
    reference = copy.deepcopy(attention)
    tokens = torch.randn(2, 64, d_model, dtype=torch.float64, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()

    output = attention(tokens)
    expected = explicit_attention(reference, reference_tokens)
    torch.testing.assert_close(output, expected, rtol=1e-9, atol=1e-10)

    # Compare gradients of every input and parameter, not just a nonzero norm.
    upstream = torch.randn_like(output)
    gradients = torch.autograd.grad(
        (output * upstream).sum(), (tokens, *attention.parameters())
    )
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(), (reference_tokens, *reference.parameters())
    )
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_uniform_attention_reads_all_squares_including_self_and_later_squares():
    attention = MultiHeadSelfAttention(4, 2)
    with torch.no_grad():
        # Zero Q/K give uniform attention. Identity V and W_O reveal its average.
        attention.qkv.weight.zero_()
        attention.qkv.bias.zero_()
        attention.qkv.weight[8:].copy_(torch.eye(4))
        attention.out_proj.weight.copy_(torch.eye(4))
        attention.out_proj.bias.zero_()
    tokens = torch.zeros(2, 64, 4)
    tokens[0, -1] = torch.tensor([64.0, 128.0, 192.0, 256.0])
    tokens[1, 0] = torch.tensor([-64.0, -128.0, -192.0, -256.0])

    expected = tokens.mean(dim=1, keepdim=True).expand_as(tokens)
    torch.testing.assert_close(attention(tokens), expected)


def test_batches_are_independent_and_square_permutations_are_preserved():
    torch.manual_seed(0)
    attention = MultiHeadSelfAttention(16, 4).double()
    # Transposed input also exercises a noncontiguous tensor.
    tokens = torch.randn(64, 2, 16, dtype=torch.float64).transpose(0, 1)
    permutation = torch.randperm(64)

    output = attention(tokens)
    separate = torch.cat([attention(tokens[i : i + 1]) for i in range(2)], dim=0)
    torch.testing.assert_close(output, separate, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(
        attention(tokens[:, permutation]), output[:, permutation], rtol=1e-9, atol=1e-10
    )


@pytest.mark.parametrize(
    "d_model,n_heads", [(0, 1), (-4, 1), (16, 0), (16, -1), (15, 4)]
)
def test_rejects_invalid_dimensions(d_model, n_heads):
    with pytest.raises(ValueError, match="positive|divisible"):
        MultiHeadSelfAttention(d_model, n_heads)


@pytest.mark.parametrize("shape", [(64, 16), (2, 16, 64), (2, 63, 16), (2, 64, 15)])
def test_rejects_incompatible_token_layout(shape):
    with pytest.raises(ValueError, match=r"expected \[batch, 64, 16\]"):
        MultiHeadSelfAttention(16, 4)(torch.zeros(shape))


def test_gab_attention_outputs_and_all_gradients_match_explicit_attention():
    torch.manual_seed(0)
    attention = MultiHeadSelfAttention(12, 3).double()
    gab = GeometricAttentionBias(12, 3, GABTemplates(5), d1=3, d2=7).double()
    reference_attention = copy.deepcopy(attention)
    reference_gab = copy.deepcopy(gab)
    tokens = torch.randn(2, 64, 12, dtype=torch.float64, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()

    output = attention(tokens, attention_bias=gab(tokens))
    expected = explicit_attention(
        reference_attention, reference_tokens, reference_gab(reference_tokens)
    )
    torch.testing.assert_close(output, expected, rtol=1e-9, atol=1e-10)
    upstream = torch.randn_like(output)
    gradients = torch.autograd.grad(
        (output * upstream).sum(), (tokens, *attention.parameters(), *gab.parameters())
    )
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(),
        (
            reference_tokens,
            *reference_attention.parameters(),
            *reference_gab.parameters(),
        ),
    )
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_zero_bias_preserves_attention_and_row_constants_cancel_in_softmax():
    torch.manual_seed(0)
    attention = MultiHeadSelfAttention(8, 2).double()
    tokens = torch.randn(2, 64, 8, dtype=torch.float64)
    zeros = torch.zeros(2, 2, 64, 64, dtype=torch.float64)
    bias = torch.randn_like(zeros)
    row_constants = torch.randn(2, 2, 64, 1, dtype=torch.float64)
    torch.testing.assert_close(attention(tokens, zeros), attention(tokens))
    torch.testing.assert_close(
        attention(tokens, bias),
        attention(tokens, bias + row_constants),
        rtol=1e-9,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "shape", [(64, 64), (1, 2, 64, 64), (2, 1, 64, 64), (2, 2, 63, 64)]
)
def test_bias_requires_explicit_batch_head_and_square_axes(shape):
    with pytest.raises(ValueError, match="expected attention_bias shape"):
        MultiHeadSelfAttention(8, 2)(torch.zeros(2, 64, 8), torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64])
def test_bias_rejects_mask_semantics(dtype):
    with pytest.raises(TypeError, match="attention_bias must be floating point"):
        MultiHeadSelfAttention(8, 2)(
            torch.zeros(2, 64, 8), torch.zeros(2, 2, 64, 64, dtype=dtype)
        )
