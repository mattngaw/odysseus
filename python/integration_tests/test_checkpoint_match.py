"""Requires: cargo build -p odysseus --bins --examples --locked --offline."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_neural_uci import Engine, stats

from neurodiktyon import ChessModel
from neurodiktyon.checkpoint import save_checkpoint

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
EXECUTABLE = TARGET / "debug/examples/match_neural"
DRIVER = ROOT / "python/examples/match_checkpoints.py"
HELPERS = runpy.run_path(str(DRIVER))


@pytest.fixture
def checkpoints(tmp_path):
    paths = []
    with torch.random.fork_rng(devices=[]):
        for seed in (12, 43):
            torch.manual_seed(seed)
            model = ChessModel(
                n_blocks=1,
                d_model=16,
                n_heads=2,
                d_ff=16,
                gab_d1=2,
                gab_d2=4,
                gab_d3=3,
                d_policy=8,
                d_value_hidden=8,
            )
            path = tmp_path / f"Model {seed}.pt"
            save_checkpoint(path, model, step=seed)
            paths.append(path)
    return paths


def run(output, white, black, *args):
    return subprocess.run(
        [
            str(EXECUTABLE),
            "--python",
            sys.executable,
            "--white",
            str(white),
            "--black",
            str(black),
            "--output",
            str(output),
            "--simulations",
            "8",
            "--max-plies",
            "4",
            *args,
        ],
        cwd=output.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("swap", [False, True])
def test_each_turn_matches_independent_uci_search_with_that_players_model(
    tmp_path, checkpoints, swap
):
    white, black = checkpoints[::-1] if swap else checkpoints
    path = tmp_path / "game.jsonl"
    # Odd opening length verifies that starting with Black doesn't swap ownership.
    result = run(path, white, black, "--moves", "e2e4")
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("checkpoint CPU FP32 model;") == 2
    events = [json.loads(line) for line in path.read_text().splitlines()]
    history = ["e2e4"]
    with (
        Engine(tmp_path, checkpoint=white) as w,
        Engine(tmp_path, checkpoint=black) as b,
    ):
        for event in events[1:-1]:
            engine = w if event["side"] == "white" else b
            engine.send("position startpos moves " + " ".join(history))
            lines = engine.search("go nodes 8")
            expected = stats(lines)
            assert event["move"] == lines[-1].split()[1]
            assert set(expected) == {row["move"] for row in event["root"]}
            for row in event["root"]:
                n, p, q = expected[row["move"]]
                assert row["visits"] == n
                assert abs(row["prior"] - p) < 1e-8
                assert abs(row["q"] - q) < 1e-7
            history.append(event["move"])
        w.send("position startpos moves " + " ".join(history))
        w.send("isready")
        assert w.until("readyok") == ["readyok"]
    assert events[-1]["completed"] is False
    assert events[-1]["winner"] is None
    assert events[-1]["played_plies"] == 4
    original = path.read_bytes()
    assert run(path, white, black).returncode != 0
    assert path.read_bytes() == original


def test_frozen_fixture_replays_all_pairs_with_balanced_colors_and_no_models(tmp_path):
    checkpoint = tmp_path / "unused.pt"
    checkpoint.write_bytes(b"not loaded at zero cap")
    output = tmp_path / "matches"
    result = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--a",
            str(checkpoint),
            "--b",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--executable",
            str(EXECUTABLE),
            "--max-plies",
            "0",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["summary"]["unfinished"] == 32
    assert manifest["summary"]["draws"] == 0
    assert manifest["summary"]["B_score_completed"] is None
    games = manifest["games"]
    assert len({g["starting_fen"] for g in games}) == 16
    for i in range(0, 32, 2):
        a, b = games[i : i + 2]
        assert a["opening_moves"] == b["opening_moves"]
        assert a["starting_fen"] == b["starting_fen"]
        assert a["white"] == b["black"] and a["black"] == b["white"]


@pytest.mark.parametrize(
    "fen,reason,winner",
    [
        ("k7/1Q6/2K5/8/8/8/8/8 b - - 0 1", "automatic_checkmate", "white"),
        ("k7/8/1QK5/8/8/8/8/8 b - - 0 1", "automatic_stalemate", None),
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "automatic_insufficient_material", None),
        ("4k3/8/8/8/8/8/8/R3K3 w - - 150 76", "automatic_seventy_five_move_rule", None),
    ],
)
def test_terminal_outcomes_precede_cap_and_bypass_workers(
    tmp_path, fen, reason, winner
):
    path = tmp_path / "terminal.jsonl"
    result = run(path, "missing.pt", "missing.pt", "--fen", fen, "--max-plies", "0")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    final = json.loads(path.read_text().splitlines()[-1])
    assert final["completed"] is True
    assert (final["reason"], final["winner"]) == (reason, winner)


def test_terminal_after_last_allowed_move_is_completed(tmp_path, checkpoints):
    path = tmp_path / "last-ply.jsonl"
    result = run(
        path,
        *checkpoints,
        "--fen",
        "4k3/8/8/8/8/8/8/R3K3 w - - 149 76",
        "--max-plies",
        "1",
    )
    assert result.returncode == 0, result.stderr
    final = json.loads(path.read_text().splitlines()[-1])
    assert final["played_plies"] == 1
    assert final["completed"] is True
    assert final["reason"] == "automatic_seventy_five_move_rule"


def test_bad_inputs_and_checkpoint_never_produce_a_result(tmp_path):
    path = tmp_path / "bad.jsonl"
    for args in [("--simulations", "0"), ("--moves", "e2e5"), ("--fen", "invalid")]:
        assert run(path, "missing.pt", "missing.pt", *args).returncode != 0
        assert not path.exists()
    failed = run(path, "missing.pt", "missing.pt")
    assert failed.returncode != 0
    assert "untrained" not in failed.stderr
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == [
        "start"
    ]


def test_scoring_does_not_award_half_a_point_to_unfinished_games():
    games = [
        {
            "completed": done,
            "winner_checkpoint": winner,
            "reason": reason,
            "played_plies": 1,
            "search_seconds": 0,
        }
        for done, winner, reason in [
            (True, "B", "mate"),
            (True, "A", "mate"),
            (True, None, "draw"),
            (False, None, "unfinished_ply_cap"),
        ]
    ]
    summary = HELPERS["summarize"](games)
    assert (
        summary["B_wins"],
        summary["draws"],
        summary["B_losses"],
        summary["unfinished"],
    ) == (1, 1, 1, 1)
    assert summary["B_points_completed"] == 1.5
    assert summary["B_score_completed"] == 0.5
