"""Requires: cargo build -p odysseus --bins --examples --locked --offline."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_neural_uci import Engine, policy_index, stats

from neurodiktyon import ChessModel
from neurodiktyon.checkpoint import save_checkpoint
from neurodiktyon.training_data import collate_examples, read_games

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
EXECUTABLE = TARGET / "debug" / "examples" / "self_play_neural"
DRAW_FEN = "4k3/8/8/8/8/8/8/R3K3 w - - 148 75"


@pytest.fixture
def checkpoint(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        model = ChessModel(
            n_blocks=2, d_model=32, n_heads=4, d_ff=32, d_policy=16, d_value_hidden=16
        )
    path = tmp_path / "Small  Network.pt"
    save_checkpoint(path, model, step=7)
    return path


def run(path, checkpoint, *args):
    return subprocess.run(
        [
            str(EXECUTABLE),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(path),
            "--python",
            sys.executable,
            "--simulations",
            "16",
            "--seed",
            "42",
            "--temperature",
            "1",
            "--max-plies",
            "2",
            *args,
        ],
        cwd=path.parent,  # Must not depend on running in the repository.
        capture_output=True,
        text=True,
        timeout=60,
    )


def played_moves(stdout):
    lines = stdout.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("Played "))
    return lines[index + 1].split()


def test_checkpoint_self_play_preserves_search_visits_history_and_python_handoff(
    tmp_path, checkpoint
):
    path = tmp_path / "game.jsonl"
    result = run(path, checkpoint, "--fen", DRAW_FEN)
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("checkpoint CPU FP32 model;") == 1
    assert "step=7; blocks=2; d_model=32;" in result.stderr
    assert "untrained" not in result.stderr
    assert "wrote 2 labeled examples" in result.stdout
    moves = played_moves(result.stdout)
    assert len(moves) == 2
    (game,) = read_games(path)
    assert game.adjudication == "automatic_seventy_five_move_rule"
    assert game.winner is None
    assert [e.side_to_move for e in game.examples] == ["white", "black"]
    batch = collate_examples(game.examples)
    assert batch.features.shape == (2, 64, 110)
    assert batch.policy_targets.shape == batch.legal_mask.shape == (2, 1858)
    assert batch.value_targets.tolist() == [[0, 1, 0], [0, 1, 0]]
    assert batch.features[0, :, 13:104].count_nonzero() == 0
    assert batch.features[1, :, 13:26].count_nonzero() == 3
    assert (
        batch.features[:, 0, 109].tolist()
        == torch.tensor([148 / 150, 149 / 150]).tolist()
    )
    torch.testing.assert_close(batch.policy_targets.sum(dim=1), torch.ones(2))
    assert (batch.policy_targets[~batch.legal_mask] == 0).all()

    # Independently replay the actual history through UCI. Both callers must
    # produce identical per-move raw visits, including every legal zero-visit slot.
    with Engine(tmp_path, checkpoint=checkpoint) as engine:
        for ply, example in enumerate(game.examples):
            history = " moves " + " ".join(moves[:ply]) if ply else ""
            engine.send(f"position fen {DRAW_FEN}{history}")
            root = stats(engine.search("go nodes 16"))
            expected = {
                policy_index(move, black=bool(ply % 2)): row[0]
                for move, row in root.items()
            }
            assert (
                dict(zip(example.legal_indices, example.visits, strict=True))
                == expected
            )
            assert example.total_visits == 16
            indices = list(example.legal_indices)
            assert batch.legal_mask[ply].sum() == len(indices)
            torch.testing.assert_close(
                batch.policy_targets[ply, indices],
                torch.tensor([n / 16 for n in example.visits]),
                rtol=0,
                atol=0,
            )
            assert moves[ply] in root
        # The final replayed position really has the announced automatic outcome.
        engine.send(f"position fen {DRAW_FEN} moves {' '.join(moves)}")
        assert engine.search("go nodes 16")[-1] == "bestmove 0000"

    replay = tmp_path / "replay.jsonl"
    repeated = run(replay, checkpoint, "--fen", DRAW_FEN)
    assert repeated.returncode == 0, repeated.stderr
    assert played_moves(repeated.stdout) == moves
    assert replay.read_bytes() == path.read_bytes()


def test_ply_limit_never_exports_a_draw_label(tmp_path, checkpoint):
    path = tmp_path / "capped.jsonl"
    result = run(path, checkpoint)
    assert result.returncode == 0, result.stderr
    assert len(played_moves(result.stdout)) == 2
    assert "Truncated: excluded 2 unlabeled roots" in result.stdout
    assert path.read_bytes() == b""
    assert list(read_games(path)) == []
    assert result.stderr.count("checkpoint CPU FP32 model;") == 1


def test_terminal_start_bypasses_model_and_output_is_never_overwritten(tmp_path):
    path = tmp_path / "terminal.jsonl"
    args = ("--fen", "k7/1Q6/2K5/8/8/8/8/8 b - - 0 1")
    result = run(path, tmp_path / "missing.pt", *args)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    (game,) = read_games(path)
    assert (game.adjudication, game.winner, game.examples) == (
        "automatic_checkmate",
        "white",
        (),
    )
    original = path.read_bytes()
    result = run(path, tmp_path / "missing.pt", *args)
    assert result.returncode != 0
    assert result.stderr.count("checkpoint CPU FP32 model;") == 0
    assert path.read_bytes() == original


def test_bad_checkpoint_fails_without_fallback_or_training_records(tmp_path):
    path = tmp_path / "failed.jsonl"
    result = run(path, tmp_path / "missing.pt")
    assert result.returncode != 0
    assert "worker startup failed:" in result.stderr
    assert "untrained" not in result.stderr
    assert path.read_bytes() == b""


@pytest.mark.parametrize(
    "args",
    [
        ("--simulations", "0"),
        ("--temperature", "NaN"),
        ("--temperature", "-1"),
        ("--max-plies", "-1"),
        ("--fen", "invalid"),
        ("--typo", "1"),
        ("--seed",),
    ],
)
def test_invalid_settings_fail_before_creating_output(tmp_path, args):
    path = tmp_path / "invalid.jsonl"
    result = run(path, tmp_path / "missing.pt", *args)
    assert result.returncode != 0
    assert not path.exists()
    assert "worker startup failed:" not in result.stderr
