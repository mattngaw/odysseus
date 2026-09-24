"""Collect complete neural self-play games under a measured position budget."""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from neurodiktyon.training_data import read_games

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_manifest(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def play_one(
    directory, checkpoint, executable, seed, index, deadline, *, max_plies=1024
):
    """Persist raw outputs and reap the entire process group on failure/timeout.

    Only fully validated completed-game JSONL is eligible for training. An
    interrupted file is retained as .incomplete, never mistaken for a game.
    """
    stem = directory / f"game-{index:04d}"
    path = stem.with_suffix(".jsonl")
    command = [
        str(executable),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(path),
        "--python",
        sys.executable,
        "--simulations",
        "32",
        "--seed",
        str(seed),
        "--temperature",
        "1",
        "--max-plies",
        str(max_plies),
    ]
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("collection deadline reached before starting a game")
    start = time.monotonic()
    with (
        stem.with_suffix(".stdout.txt").open("x") as stdout,
        stem.with_suffix(".stderr.txt").open("x") as stderr,
    ):
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=remaining)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()
            if path.exists():
                path.rename(stem.with_suffix(".incomplete"))
            raise
    if returncode != 0:
        raise RuntimeError(
            f"self-play failed ({returncode}); inspect {stem}.stderr.txt"
        )
    games = list(read_games(path))
    stdout = stem.with_suffix(".stdout.txt").read_text()
    lines = stdout.splitlines()
    move_line = next(i for i, line in enumerate(lines) if line.startswith("Played "))
    moves = lines[move_line + 1].split()
    if len(games) > 1:
        raise ValueError("expected at most one completed game")
    count = len(games[0].examples) if games else 0
    if games and (not count or count != len(moves)):
        raise ValueError("completed-game example count disagrees with played plies")
    if not games and (path.stat().st_size != 0 or "Truncated:" not in stdout):
        raise ValueError("empty output without an explicit truncation")
    if any(example.total_visits != 32 for game in games for example in game.examples):
        raise ValueError("unexpected root simulation budget")
    stderr = stem.with_suffix(".stderr.txt").read_text()
    if stderr.count("checkpoint CPU FP32 model;") != 1 or "untrained" in stderr:
        raise ValueError("expected exactly one checkpoint worker per game")
    return {
        "id": index,
        "seed": seed,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "command": command,
        "cwd": str(ROOT),
        "elapsed_seconds": time.monotonic() - start,
        "plies": len(moves),
        "moves": moves,
        "final_fen": next(
            line.removeprefix("Final FEN: ")
            for line in lines
            if line.startswith("Final FEN: ")
        ),
        "examples": count,
        "completed": bool(games),
        "outcome": games[0].adjudication if games else "truncated",
        "winner": games[0].winner if games else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-seconds", type=float, default=900)
    parser.add_argument("--total-seconds", type=float, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not (
        math.isfinite(args.collection_seconds)
        and 0 < args.collection_seconds < args.total_seconds
    ):
        parser.error("require 0 < collection-seconds < total-seconds")
    if not math.isfinite(args.total_seconds) or not 0 <= args.seed < (1 << 64) - 100000:
        parser.error("invalid total-seconds or seed")
    checkpoint = args.checkpoint.resolve()
    executable = ROOT / "target/release/examples/self_play_neural"
    parent_hash, executable_hash = sha256(checkpoint), sha256(executable)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started_wall, started = time.time(), time.monotonic()
    # Preserve at least a quarter of the total stop budget for training/verification.
    collection_deadline = started + min(
        args.collection_seconds * 2, args.total_seconds * 0.75
    )
    report = {
        "version": 1,
        "purpose": "one A-to-B self-play learning smoke run",
        "command": [sys.executable, *sys.argv],
        "cwd": str(ROOT),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": parent_hash,
        "executable_sha256": executable_hash,
        "started_unix": started_wall,
        "deadline_unix": started_wall + args.total_seconds,
        "requested_collection_seconds": args.collection_seconds,
        "total_budget_seconds": args.total_seconds,
        "pilot_games": 4,
        "minimum_completed_games": 16,
        "status": "collecting",
        "games": [],
    }
    manifest = directory / "manifest.json"
    save_manifest(manifest, report)
    try:
        target = None
        while True:
            game = play_one(
                directory,
                checkpoint,
                executable,
                args.seed + len(report["games"]),
                len(report["games"]),
                collection_deadline,
            )
            report["games"].append(game)
            report["elapsed_seconds"] = time.monotonic() - started
            report["completed_games"] = sum(g["completed"] for g in report["games"])
            report["labeled_positions"] = sum(g["examples"] for g in report["games"])
            report["played_plies"] = sum(g["plies"] for g in report["games"])
            print(
                f"game={game['id']} seed={game['seed']} outcome={game['outcome']} plies={game['plies']} labeled={report['labeled_positions']} completed={report['completed_games']} elapsed={report['elapsed_seconds']:.1f}s",
                flush=True,
            )
            if len(report["games"]) == 4:
                if report["labeled_positions"] == 0:
                    raise RuntimeError(
                        "no completed pilot games; inspect completion before training"
                    )
                rate = report["labeled_positions"] / report["elapsed_seconds"]
                target = math.ceil(rate * args.collection_seconds)
                report["pilot_labeled_positions_per_second"] = rate
                report["frozen_target_positions"] = target
                print(
                    f"Frozen target: {target} labeled positions at pilot rate {rate:.2f}/s",
                    flush=True,
                )
            save_manifest(manifest, report)
            if (
                target is not None
                and report["labeled_positions"] >= target
                and report["completed_games"] >= 16
            ):
                break
        if sha256(checkpoint) != parent_hash:
            raise RuntimeError("parent checkpoint changed during collection")
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save_manifest(manifest, report)


if __name__ == "__main__":
    main()
