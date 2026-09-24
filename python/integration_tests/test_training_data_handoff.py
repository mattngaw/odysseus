"""Requires: cargo build -p odysseus --example self_play_export --locked --offline."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from neurodiktyon import ModelOutput, training_loss
from neurodiktyon.training_data import collate_examples, read_games


def test_real_self_play_export_loads_without_losing_inputs_or_visit_information(
    tmp_path,
):
    root = Path(__file__).resolve().parents[2]
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
    executable = target / "debug" / "examples" / "self_play_export"
    if sys.platform == "win32":
        executable = executable.with_suffix(".exe")
    path = tmp_path / "self-play.jsonl"
    result = subprocess.run(
        [str(executable), str(path)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert "Truncated opening: excluded 2 unlabeled roots" in result.stdout
    games = list(read_games(path))
    assert [g.adjudication for g in games] == [
        "automatic_checkmate",
        "automatic_checkmate",
        "automatic_seventy_five_move_rule",
        "automatic_checkmate",
    ]
    assert [g.winner for g in games] == ["white", "black", None, "white"]
    assert [len(g.examples) for g in games] == [1, 1, 2, 2]
    examples = [e for game in games for e in game.examples]
    raw_examples = [
        e
        for line in path.read_text().splitlines()
        for e in json.loads(line)["examples"]
    ]
    for example, raw in zip(examples, raw_examples, strict=True):
        torch.testing.assert_close(
            example.features, torch.tensor(raw["input"]), rtol=0, atol=0
        )
        assert example.legal_indices == tuple(e["index"] for e in raw["policy"])
        assert example.visits == tuple(e["visits"] for e in raw["policy"])
        assert example.total_visits == raw["total_visits"] == 32
    batch = collate_examples(examples)
    assert batch.features.shape == (6, 64, 110)
    assert batch.value_targets.tolist() == [
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]
    # Distinct inputs allow a fixed-batch memorization check with all W/D/L labels.
    assert torch.unique(batch.features.flatten(1), dim=0).shape[0] == 6
    assert batch.features[0, 41, 4].item() == 1  # Our queen on relative b6.
    assert batch.features[0, 42, 5].item() == 1  # Our king on relative c6.
    assert batch.features[0, 56, 11].item() == 1  # Their king on relative a8.
    assert batch.features[1, 46, 4].item() == 1  # Black Qg3 -> our relative g6.
    assert batch.features[1, 45, 5].item() == 1  # Black Kf3 -> our relative f6.
    assert batch.features[1, 63, 11].item() == 1  # White Kh1 -> their relative h8.
    assert batch.features[2, 0, 109].item() == torch.tensor(148 / 150).item()
    assert batch.features[3, 0, 109].item() == torch.tensor(149 / 150).item()
    assert batch.features[2, :, 13:104].count_nonzero().item() == 0
    assert batch.features[3, :, 13:26].count_nonzero().item() == 3
    for row, example in enumerate(examples):
        indices = list(example.legal_indices)
        assert batch.legal_mask[row].sum().item() == len(indices)
        expected = torch.tensor([n / example.total_visits for n in example.visits])
        torch.testing.assert_close(
            batch.policy_targets[row, indices], expected, rtol=0, atol=0
        )
    assert (batch.legal_mask & (batch.policy_targets == 0)).any()
    assert (batch.policy_targets[~batch.legal_mask] == 0).all()
    torch.testing.assert_close(batch.policy_targets.sum(dim=1), torch.ones(6))

    # The handoff plugs directly into the existing loss interface, including masks.
    output = ModelOutput(
        torch.zeros(6, 1858, requires_grad=True), torch.zeros(6, 3, requires_grad=True)
    )
    losses = training_loss(
        output, batch.policy_targets, batch.value_targets, batch.legal_mask
    )
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert (output.policy_logits.grad[~batch.legal_mask] == 0).all()
