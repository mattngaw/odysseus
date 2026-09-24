import pytest
import torch

from neurodiktyon import SquareEmbedding


def test_preserves_square_order_and_uses_shared_weights():
    torch.manual_seed(0)
    embedding = SquareEmbedding(32)
    features = torch.randn(2, 64, 110)
    permutation = torch.randperm(64)

    tokens = embedding(features)

    assert tokens.shape == (2, 64, 32)
    torch.testing.assert_close(
        embedding(features[:, permutation]), tokens[:, permutation]
    )


def test_changing_one_square_does_not_change_other_tokens():
    torch.manual_seed(0)
    embedding = SquareEmbedding(32)
    features = torch.randn(2, 64, 110)
    changed = features.clone()
    changed[0, 12, 0] += 1.0

    tokens = embedding(features)
    changed_tokens = embedding(changed)
    changed_squares = (tokens != changed_tokens).any(dim=-1)

    assert changed_squares.sum().item() == 1
    assert changed_squares[0, 12]


def test_selected_token_has_local_input_gradients_and_trainable_parameters():
    torch.manual_seed(0)
    embedding = SquareEmbedding(32)
    features = torch.randn(2, 64, 110, requires_grad=True)

    embedding(features)[0, 12].square().sum().backward()

    assert features.grad is not None
    affected_squares = (features.grad != 0).any(dim=-1)
    assert affected_squares.sum().item() == 1
    assert affected_squares[0, 12]
    for parameter in embedding.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(64, 110), (2, 110, 64), (2, 63, 110), (2, 64, 109)])
def test_rejects_incompatible_encoder_layout(shape):
    with pytest.raises(ValueError, match=r"expected \[batch, 64, 110\]"):
        SquareEmbedding(32)(torch.zeros(shape))


@pytest.mark.parametrize("d_model", [0, -1])
def test_rejects_nonpositive_hidden_dimension(d_model):
    with pytest.raises(ValueError, match="d_model must be positive"):
        SquareEmbedding(d_model)
