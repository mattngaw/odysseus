from dataclasses import asdict

import pytest
import torch

from neurodiktyon import ChessModel
from neurodiktyon.checkpoint import load_checkpoint, save_checkpoint


@pytest.fixture
def model():
    torch.manual_seed(7)
    return ChessModel(
        n_blocks=2,
        d_model=8,
        n_heads=2,
        d_ff=12,
        gab_d1=2,
        gab_d2=4,
        gab_d3=3,
        d_policy=8,
        d_value_hidden=4,
    )


def test_snapshot_preserves_architecture_all_state_sharing_predictions_and_rng(
    model, tmp_path
):
    features = torch.randn(2, 64, 110)
    model.eval()
    with torch.inference_mode():
        original = model(features)
    rng = torch.get_rng_state().clone()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, step=11)
    loaded = load_checkpoint(path)
    assert torch.equal(rng, torch.get_rng_state())
    assert loaded.step == 11
    assert loaded.model.config == model.config
    assert not loaded.model.training
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], tensor)
    for block in loaded.model.trunk.blocks:
        assert block.gab.templates is loaded.model.trunk.templates
    with torch.inference_mode():
        restored = loaded.model(features)
    for a, b in zip(original, restored, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    payload = torch.load(path, weights_only=True)
    assert payload["model_config"] == asdict(model.config)
    assert payload["value_order"] == ["win", "draw", "loss"]
    assert all(t.device.type == "cpu" for t in payload["model_state"].values())
    assert "optimizer_state" not in payload
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, step=12)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("version", 2),
        ("version", True),
        ("architecture", "other"),
        ("input_encoding", "wrong-planes"),
        ("policy_vocabulary", "wrong-order"),
        ("value_order", ["loss", "draw", "win"]),
        ("dtype", "bfloat16"),
        ("step", -1),
        ("step", True),
    ],
)
def test_incompatible_metadata_is_rejected(model, tmp_path, key, value):
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, step=0)
    payload = torch.load(path, weights_only=True)
    payload[key] = value
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_tensor",
        "wrong_shape",
        "wrong_dtype",
        "nonfinite",
        "vocabulary",
        "shared_templates",
        "missing_config",
        "invalid_config",
    ],
)
def test_inconsistent_state_cannot_silently_change_model_semantics(
    model, tmp_path, damage
):
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, step=0)
    payload = torch.load(path, weights_only=True)
    state = payload["model_state"]
    name = "trunk.embedding.projection.weight"
    if damage == "missing_tensor":
        del state[name]
    elif damage == "wrong_shape":
        state[name] = state[name][:1]
    elif damage == "wrong_dtype":
        state[name] = state[name].double()
    elif damage == "nonfinite":
        state[name][0, 0] = float("nan")
    elif damage == "vocabulary":
        key = next(key for key in state if key.endswith("base_indices"))
        state[key] = state[key].roll(1)
    elif damage == "shared_templates":
        key = "trunk.blocks.0.gab.templates.projection.weight"
        # Break storage sharing too, otherwise the corruption changes all aliases.
        state[key] = state[key].clone() + 1
    elif damage == "missing_config":
        del payload["model_config"]["gab_d3"]
    else:
        payload["model_config"]["n_heads"] = 3
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_invalid_snapshots_fail_before_creating_a_file(model, tmp_path):
    path = tmp_path / "bad.pt"
    with pytest.raises(ValueError, match="step"):
        save_checkpoint(path, model, step=-1)
    assert not path.exists()
    with pytest.raises(ValueError, match="dtype"):
        save_checkpoint(path, model.double(), step=0)
    assert not path.exists()
    model.float()
    with torch.no_grad():
        next(model.parameters()).fill_(float("inf"))
    with pytest.raises(ValueError, match="non-finite"):
        save_checkpoint(path, model, step=0)
    assert not path.exists()
