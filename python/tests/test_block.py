import pytest
import torch
from torch import nn

from neurodiktyon import GABTemplates, TransformerBlock


def small_block():
    return TransformerBlock(12, 3, 19, GABTemplates(5), gab_d1=3, gab_d2=7).double()


def test_zero_branches_preserve_inputs_and_gradients_exactly():
    torch.manual_seed(0)
    block = small_block()
    with torch.no_grad():
        for projection in (block.attention.out_proj, block.ffn.out_proj):
            projection.weight.zero_()
            projection.bias.zero_()
    tokens = torch.randn(2, 64, 12, dtype=torch.float64, requires_grad=True)
    output = block(tokens)
    assert torch.equal(output, tokens)

    upstream = torch.randn_like(output)
    gradient = torch.autograd.grad(output, tokens, grad_outputs=upstream)[0]
    assert torch.equal(gradient, upstream)


def test_ffn_sees_normalized_updated_residual_with_its_own_affine_parameters():
    torch.manual_seed(0)
    block = small_block()
    attention_update = torch.linspace(-2, 3, 12, dtype=torch.float64)
    scale = torch.linspace(0.5, 1.5, 12, dtype=torch.float64)
    offset = torch.linspace(-0.3, 0.7, 12, dtype=torch.float64)
    with torch.no_grad():
        # Constant attention output plus an identity FFN isolates the second norm.
        block.attention.out_proj.weight.zero_()
        block.attention.out_proj.bias.copy_(attention_update)
        block.norm_ffn.weight.copy_(scale)
        block.norm_ffn.bias.copy_(offset)
    block.ffn = nn.Identity()
    tokens = torch.randn(2, 64, 12, dtype=torch.float64)
    updated = tokens + attention_update
    centered = updated - updated.mean(dim=-1, keepdim=True)
    normalized = centered / (centered.square().mean(dim=-1, keepdim=True) + 1e-5).sqrt()

    torch.testing.assert_close(
        block(tokens), updated + normalized * scale + offset, rtol=1e-9, atol=1e-10
    )


def test_pre_norm_branches_ignore_per_square_feature_offsets():
    torch.manual_seed(0)
    block = small_block()
    tokens = torch.randn(2, 64, 12, dtype=torch.float64)
    # A scalar offset shared by a square's features is removed by LayerNorm.
    # It must survive only on the residual path, including when GAB is active.
    offsets = torch.randn(2, 64, 1, dtype=torch.float64) * 3
    torch.testing.assert_close(
        block(tokens + offsets), block(tokens) + offsets, rtol=1e-9, atol=1e-10
    )


def test_gradients_reach_inputs_and_every_component_including_shared_templates():
    torch.manual_seed(0)
    block = small_block()
    tokens = torch.randn(2, 64, 12, dtype=torch.float64, requires_grad=True)
    output = block(tokens)
    (output * torch.randn_like(output)).sum().backward()

    for name, tensor in [("input", tokens), *block.named_parameters()]:
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum() > 0, name


def test_batch_items_are_independent_with_noncontiguous_input():
    torch.manual_seed(0)
    block = small_block()
    tokens = torch.randn(64, 2, 12, dtype=torch.float64).transpose(0, 1)
    assert not tokens.is_contiguous()
    separate = torch.cat([block(tokens[i : i + 1]) for i in range(2)], dim=0)
    torch.testing.assert_close(block(tokens), separate, rtol=1e-9, atol=1e-10)


def test_only_template_parameters_are_shared_between_blocks():
    templates = GABTemplates(5)
    first = TransformerBlock(12, 3, 19, templates, gab_d1=3, gab_d2=7)
    second = TransformerBlock(12, 3, 19, templates, gab_d1=3, gab_d2=7)
    first_ids = {id(p) for p in first.parameters()}
    second_ids = {id(p) for p in second.parameters()}
    template_ids = {id(p) for p in templates.parameters()}

    assert first.gab.templates is second.gab.templates is templates
    assert first_ids & second_ids == template_ids
    assert {id(p) for p in first.norm_attention.parameters()}.isdisjoint(
        id(p) for p in first.norm_ffn.parameters()
    )
    combined = nn.ModuleList([first, second])
    assert len(list(combined.parameters())) == len(first_ids | second_ids)


@pytest.mark.parametrize("shape", [(64, 12), (2, 12, 64), (2, 63, 12), (2, 64, 11)])
def test_rejects_incompatible_token_layout(shape):
    with pytest.raises(ValueError, match=r"expected \[batch, 64, 12\]"):
        small_block()(torch.zeros(shape, dtype=torch.float64))
