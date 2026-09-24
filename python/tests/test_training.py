import pytest
import torch

from neurodiktyon import ChessModel
from neurodiktyon.checkpoint import load_checkpoint, save_checkpoint
from neurodiktyon.training import train_fixed_batch
from neurodiktyon.training_data import TrainingBatch


@pytest.fixture
def model_and_batch():
    torch.manual_seed(0)
    model = ChessModel(
        n_blocks=1,
        d_model=8,
        n_heads=2,
        d_ff=12,
        gab_d1=2,
        gab_d2=4,
        gab_d3=3,
        d_policy=8,
        d_value_hidden=8,
    )
    # Synthetic tensors isolate the optimizer; the integration demo uses real games.
    targets = torch.zeros(3, 1858)
    targets[:, :3] = torch.tensor([0.6, 0.3, 0.1])
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[:, :4] = True
    return model, TrainingBatch(torch.randn(3, 64, 110), targets, mask, torch.eye(3))


def test_cpu_fixed_batch_updates_both_heads_and_trunk_then_reloads(
    model_and_batch, tmp_path
):
    model, batch = model_and_batch
    original_batch = tuple(t.clone() for t in batch)
    result = train_fixed_batch(model, batch, steps=30, learning_rate=0.01)
    assert [row.step for row in result.losses] == list(range(31))
    assert result.gradient_checks == 30
    assert result.losses[-1].policy < result.losses[0].policy
    assert result.losses[-1].value < result.losses[0].value
    assert all(change > 0 for change in result.parameter_max_changes.values())
    for row in result.losses:
        assert row.total == pytest.approx(row.policy + row.value)
    for original, current in zip(original_batch, batch, strict=True):
        assert torch.equal(original, current)
    path = tmp_path / "trained.pt"
    save_checkpoint(path, model, step=30)
    loaded = load_checkpoint(path)
    assert loaded.step == 30
    with torch.inference_mode():
        for a, b in zip(
            model.eval()(batch.features), loaded.model(batch.features), strict=True
        ):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"steps": 0}, "steps"),
        ({"steps": True}, "steps"),
        ({"steps": 1, "learning_rate": 0}, "learning_rate"),
        ({"steps": 1, "learning_rate": float("nan")}, "learning_rate"),
        ({"steps": 1, "learning_rate": float("inf")}, "learning_rate"),
    ],
)
def test_invalid_settings_do_not_mutate_parameters(model_and_batch, options, message):
    model, batch = model_and_batch
    before = {name: p.clone() for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match=message):
        train_fixed_batch(model, batch, **options)
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())


def test_nonfinite_gradient_stops_before_optimizer_update(model_and_batch):
    model, batch = model_and_batch
    before = {name: p.clone() for name, p in model.named_parameters()}
    handle = model.value_head.out_proj.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    try:
        with pytest.raises(FloatingPointError, match="non-finite gradients at step 0"):
            train_fixed_batch(model, batch, steps=2)
    finally:
        handle.remove()
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())


def test_disconnected_head_is_caught_before_optimizer_update(model_and_batch):
    model, batch = model_and_batch
    before = {name: p.clone() for name, p in model.named_parameters()}
    handle = model.value_head.register_forward_hook(
        lambda module, args, output: output.detach().requires_grad_()
    )
    try:
        with pytest.raises(RuntimeError, match="missing gradients at step 0"):
            train_fixed_batch(model, batch, steps=1)
    finally:
        handle.remove()
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())
