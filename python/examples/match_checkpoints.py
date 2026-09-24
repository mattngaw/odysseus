"""Run fixed, color-swapped checkpoint matches through the Rust match example."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPENINGS = Path(__file__).with_name("fixtures") / "match_openings.json"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def schedule(openings):
    if not isinstance(openings, list) or not openings:
        raise ValueError("expected a nonempty opening list")
    names, lines, games = set(), set(), []
    for pair, opening in enumerate(openings):
        name, moves = opening["name"], opening["moves"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("opening names must be distinct nonempty strings")
        if not isinstance(moves, list) or not all(isinstance(m, str) for m in moves):
            raise ValueError("opening moves must be a list of UCI strings")
        if tuple(moves) in lines:
            raise ValueError("duplicate opening move sequence")
        names.add(name)
        lines.add(tuple(moves))
        # Alternate which checkpoint plays White first; every pair contains both.
        for white in ("A", "B") if pair % 2 == 0 else ("B", "A"):
            games.append(
                {
                    "id": len(games),
                    "pair": pair,
                    "opening": name,
                    "opening_moves": moves,
                    "white": white,
                    "black": "B" if white == "A" else "A",
                }
            )
    return games


def read_result(path, spec, simulations, max_plies):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if (
        len(events) < 2
        or events[0].get("event") != "start"
        or events[-1].get("event") != "result"
        or any(e.get("event") != "move" for e in events[1:-1])
    ):
        raise ValueError("missing or invalid match event sequence")
    start, result = events[0], events[-1]
    moves = events[1:-1]
    if start["opening_moves"] != spec["opening_moves"]:
        raise ValueError("opening mismatch")
    if result["played_plies"] != len(moves) or len(moves) > max_plies:
        raise ValueError("incorrect played-ply count")
    for ply, event in enumerate(moves):
        root = event["root"]
        expected_side = (
            "white" if (len(spec["opening_moves"]) + ply) % 2 == 0 else "black"
        )
        if event["ply"] != ply or event["side"] != expected_side:
            raise ValueError("incorrect ply/side sequence")
        if (
            event["simulations"] != simulations
            or sum(r["visits"] for r in root) != simulations
        ):
            raise ValueError("unexpected simulation budget")
        # Python max, like Pyxis, keeps the first item on equal visit counts.
        if event["move"] != max(root, key=lambda r: r["visits"])["move"]:
            raise ValueError("move was not the stable greedy visit choice")
    if not result["completed"]:
        if (
            result["reason"] != "unfinished_ply_cap"
            or result["winner"] is not None
            or len(moves) != max_plies
        ):
            raise ValueError("invalid unfinished result")
    elif result["reason"] == "unfinished_ply_cap":
        raise ValueError("ply cap cannot be a completed outcome")
    winner = result["winner"]
    return {
        **spec,
        **result,
        "winner_checkpoint": spec[winner] if winner else None,
        "starting_fen": start["starting_fen"],
        "moves": [e["move"] for e in moves],
        "search_seconds": sum(e["search_seconds"] for e in moves),
        "path": str(path),
        "sha256": sha256(path),
    }


def summarize(games):
    completed = [g for g in games if g["completed"]]
    wins = sum(g["winner_checkpoint"] == "B" for g in completed)
    losses = sum(g["winner_checkpoint"] == "A" for g in completed)
    draws = len(completed) - wins - losses
    return {
        "games": len(games),
        "completed": len(completed),
        "B_wins": wins,
        "draws": draws,
        "B_losses": losses,
        "unfinished": len(games) - len(completed),
        "B_points_completed": wins + draws / 2,
        "B_score_completed": (wins + draws / 2) / len(completed) if completed else None,
        "outcomes": dict(Counter(g["reason"] for g in games)),
        "played_plies": sum(g["played_plies"] for g in games),
        "search_seconds": sum(g["search_seconds"] for g in games),
    }


def run_process(command, stem, timeout):
    started = time.monotonic()
    with (
        stem.with_suffix(".stdout.txt").open("x") as stdout,
        stem.with_suffix(".stderr.txt").open("x") as stderr,
    ):
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(
                    f"match process failed ({code}); inspect {stem}.stderr.txt"
                )
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()
            raise
    return time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openings", type=Path, default=OPENINGS)
    parser.add_argument(
        "--executable", type=Path, default=ROOT / "target/release/examples/match_neural"
    )
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=1024)
    parser.add_argument("--total-seconds", type=float, default=1800)
    args = parser.parse_args()
    if not 0 < args.simulations < 2**32 or not 0 <= args.max_plies < 2**32:
        parser.error("require positive u32 simulations and nonnegative u32 max-plies")
    if not math.isfinite(args.total_seconds) or args.total_seconds <= 0:
        parser.error("total-seconds must be finite and positive")
    openings = json.loads(args.openings.read_text())
    plan = schedule(openings)
    checkpoints = {"A": args.a.resolve(), "B": args.b.resolve()}
    executable = args.executable.resolve()
    hashes = {name: sha256(path) for name, path in checkpoints.items()}
    executable_hash = sha256(executable)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "openings.json").write_text(json.dumps(openings, indent=2) + "\n")
    started = time.monotonic()
    report = {
        "format": "odysseus.paired_match",
        "version": 1,
        "status": "running",
        "command": [sys.executable, *sys.argv],
        "cwd": str(ROOT),
        "started_unix": time.time(),
        "checkpoints": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in checkpoints.items()
        },
        "executable": str(executable),
        "executable_sha256": executable_hash,
        "openings_sha256": sha256(directory / "openings.json"),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "torch": importlib.metadata.version("torch"),
            "device": "cpu",
            "dtype": "float32",
            "threads_per_worker": 1,
        },
        "settings": {
            "simulations": args.simulations,
            "max_additional_plies": args.max_plies,
            "exploration": 1.0,
            "temperature": 0,
            "root_noise": False,
            "total_seconds": args.total_seconds,
        },
        "schedule": plan,
        "games": [],
    }
    manifest = directory / "manifest.json"
    save(manifest, report)
    try:
        for spec in plan:
            remaining = args.total_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("match deadline reached")
            stem = directory / f"game-{spec['id']:02d}"
            path = stem.with_suffix(".jsonl")
            command = [
                str(executable),
                "--python",
                sys.executable,
                "--white",
                str(checkpoints[spec["white"]]),
                "--black",
                str(checkpoints[spec["black"]]),
                "--output",
                str(path),
                "--moves",
                " ".join(spec["opening_moves"]),
                "--simulations",
                str(args.simulations),
                "--max-plies",
                str(args.max_plies),
            ]
            wall_seconds = run_process(command, stem, remaining)
            game = read_result(path, spec, args.simulations, args.max_plies)
            game.update(command=command, cwd=str(ROOT), wall_seconds=wall_seconds)
            stderr = stem.with_suffix(".stderr.txt").read_text()
            used_sides = min(game["played_plies"], 2)
            if (
                stderr.count("checkpoint CPU FP32 model;") != used_sides
                or "untrained" in stderr
            ):
                raise ValueError(
                    "expected one persistent checkpoint worker per playing side"
                )
            report["games"].append(game)
            report["summary"] = summarize(report["games"])
            report["elapsed_seconds"] = time.monotonic() - started
            save(manifest, report)
            print(
                f"game={spec['id'] + 1}/{len(plan)} opening={spec['opening']} white={spec['white']} "
                f"winner={game['winner_checkpoint']} reason={game['reason']} plies={game['played_plies']} "
                f"B W/D/L={report['summary']['B_wins']}/{report['summary']['draws']}/{report['summary']['B_losses']} "
                f"unfinished={report['summary']['unfinished']} elapsed={report['elapsed_seconds']:.1f}s",
                flush=True,
            )
        if (
            hashes != {name: sha256(path) for name, path in checkpoints.items()}
            or sha256(executable) != executable_hash
        ):
            raise RuntimeError("checkpoint or executable changed during the match")
        report["checkpoints_unchanged"] = True
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(manifest, report)


if __name__ == "__main__":
    main()
