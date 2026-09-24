"""Checkpoint warm start -> real recorded-game batches -> portable checkpoint."""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from neurodiktyon import ChessModel
from neurodiktyon.checkpoint import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))


@pytest.fixture
def collection(tmp_path):
    torch.manual_seed(31)
    checkpoint = tmp_path / "parent.pt"
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
    save_checkpoint(checkpoint, model, step=17)
    raw = tmp_path / "raw.jsonl"
    subprocess.run(
        [str(TARGET / "debug/examples/self_play_export"), str(raw)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    records = []
    for index, line in enumerate(raw.read_text().splitlines()):
        game = json.loads(line)
        path = tmp_path / f"game-{index}.jsonl"
        path.write_text(line + "\n")
        records.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "completed": True,
                "examples": len(game["examples"]),
                "outcome": game["adjudication"]["kind"],
                "winner": game["adjudication"].get("winner"),
            }
        )
    # Capped games must stay outside both partitions.
    capped = tmp_path / "capped.jsonl"
    capped.touch()
    records.append({"path": str(capped), "completed": False})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "deadline_unix": time.time() + 300,
                "minimum_completed_games": 2,
                "games": records,
            }
        )
    )
    return manifest, checkpoint


def run(manifest, output, *args):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "python/examples/train_self_play.py"),
            "--collection",
            str(manifest),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            *args,
        ],
        cwd=manifest.parent,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("fp32", [False, True])
def test_training_cli_preserves_parent_config_split_and_exact_exposure(
    tmp_path, collection, fp32
):
    manifest, parent = collection
    parent_bytes = parent.read_bytes()
    output = tmp_path / "trained"
    process = run(manifest, output, *(("--fp32",) if fp32 else ()))
    assert process.returncode == 0, process.stderr
    report = json.loads((output / "metrics.json").read_text())
    assert report["status"] == "complete"
    assert report["compute"] == ("fp32" if fp32 else "bf16 autocast")
    train = {game["path"] for game in report["train_games"]}
    held = {game["path"] for game in report["held_out_games"]}
    assert not train & held and len(train) == 3 and len(held) == 1
    assert not any("capped" in path for path in train | held)
    assert report["samples_processed"] == report["train_positions"] * 2
    assert report["sample_reuse_ratio"] == 2
    assert report["steps"] == 2 * math.ceil(report["train_positions"] / 2)
    assert report["checkpoint_step"] == 17 + report["steps"]
    assert report["checkpoint_state_exact"]
    assert report["reload_fp32_max_absolute_errors"] == [0, 0]
    original, learned = load_checkpoint(parent), load_checkpoint(output / "model.pt")
    assert learned.model.config == original.model.config
    assert learned.step == report["checkpoint_step"]
    assert any(
        not torch.equal(p, q)
        for p, q in zip(
            original.model.parameters(), learned.model.parameters(), strict=True
        )
    )
    assert all(p.dtype == torch.float32 for p in learned.model.parameters())
    assert parent.read_bytes() == parent_bytes
    assert not (output / "model.pt.partial").exists()


@pytest.mark.parametrize("damage", ["parent", "game", "unfinished", "expired"])
def test_changed_inputs_or_unfinished_collection_never_produce_checkpoint(
    tmp_path, collection, damage
):
    manifest, parent = collection
    payload = json.loads(manifest.read_text())
    if damage == "parent":
        parent.write_bytes(b"changed checkpoint")
    elif damage == "game":
        Path(payload["games"][0]["path"]).write_text("corrupt record")
    elif damage == "unfinished":
        payload["status"] = "collecting"
    elif damage == "expired":
        payload["deadline_unix"] = time.time() - 1
    manifest.write_text(json.dumps(payload))
    output = tmp_path / "failed"
    process = run(manifest, output)
    assert process.returncode != 0
    assert not (output / "model.pt").exists()
