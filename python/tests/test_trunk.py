import copy

import pytest
import torch

from neurodiktyon import TransformerTrunk


def small_trunk():
    return TransformerTrunk(
        n_blocks=3, d_model=12, n_heads=3, d_ff=19, gab_d1=3, gab_d2=7, gab_d3=5
    ).double()


def test_only_template_parameters_are_shared_across_the_stack():
    trunk = small_trunk()
    template_ids = {id(p) for p in trunk.templates.parameters()}
    seen = {id(p) for p in trunk.embedding.parameters()} | template_ids
    assert len(trunk.blocks) == 3
    for block in trunk.blocks:
        assert block.gab.templates is trunk.templates
        block_ids = {id(p) for p in block.parameters()}
        assert block_ids & seen == template_ids
        seen |= block_ids
    final_norm_ids = {id(p) for p in trunk.final_norm.parameters()}
    assert seen.isdisjoint(final_norm_ids)
    assert {id(p) for p in trunk.parameters()} == seen | final_norm_ids


def test_all_parameters_receive_gradients_and_shared_bank_accumulates_every_use():
    torch.manual_seed(0)
    trunk = small_trunk()
    reference = copy.deepcopy(trunk)
    # Untie equal-valued template banks to measure each use's gradient separately.
    for block in reference.blocks:
        block.gab.templates = copy.deepcopy(reference.templates)

    features = torch.randn(2, 64, 110, dtype=torch.float64, requires_grad=True)
    reference_features = features.detach().clone().requires_grad_()
    output = trunk(features)
    expected = reference(reference_features)
    assert output.shape == (2, 64, 12)
    torch.testing.assert_close(output, expected, rtol=1e-9, atol=1e-10)

    upstream = torch.randn_like(output)
    (output * upstream).sum().backward()
    (expected * upstream).sum().backward()
    for name, tensor in [("input", features), *trunk.named_parameters()]:
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum() > 0, name
    torch.testing.assert_close(
        features.grad, reference_features.grad, rtol=1e-9, atol=1e-10
    )
    per_use_gradients = [
        block.gab.templates.projection.weight.grad for block in reference.blocks
    ]
    assert all(gradient is not None for gradient in per_use_gradients)
    torch.testing.assert_close(
        trunk.templates.projection.weight.grad,
        torch.stack(per_use_gradients).sum(dim=0),
        rtol=1e-9,
        atol=1e-10,
    )


def test_final_norm_applies_learned_affine_map_after_all_blocks():
    torch.manual_seed(0)
    trunk = small_trunk()
    offset = torch.linspace(-2, 3, 12, dtype=torch.float64)
    with torch.no_grad():
        trunk.final_norm.weight.zero_()
        trunk.final_norm.bias.copy_(offset)
    features = torch.randn(2, 64, 110, dtype=torch.float64)
    # A zero scale leaves only the learned offset at every square and batch item.
    torch.testing.assert_close(trunk(features), offset.expand(2, 64, 12))


def test_batch_items_remain_independent_with_noncontiguous_features():
    torch.manual_seed(0)
    trunk = small_trunk()
    features = torch.randn(64, 2, 110, dtype=torch.float64).transpose(0, 1)
    assert not features.is_contiguous()
    separate = torch.cat([trunk(features[i : i + 1]) for i in range(2)], dim=0)
    torch.testing.assert_close(trunk(features), separate, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("n_blocks", [0, -1])
def test_rejects_nonpositive_block_count(n_blocks):
    with pytest.raises(ValueError, match="n_blocks must be positive"):
        TransformerTrunk(n_blocks=n_blocks)


def test_requires_encoded_features_not_hidden_tokens():
    with pytest.raises(ValueError, match=r"expected \[batch, 64, 110\]"):
        small_trunk()(torch.zeros(2, 64, 12, dtype=torch.float64))
