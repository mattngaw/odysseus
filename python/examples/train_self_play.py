"""Warm-start one model from a frozen self-play collection and save checkpoint B."""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from neurodiktyon.checkpoint import load_checkpoint, save_checkpoint
from neurodiktyon.minibatch_training import split_game_indices, train_minibatches
from neurodiktyon.training_data import collate_examples, read_games


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        type=Path,
        required=True,
        help="completed collection manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="disable BF16 autocast for a reference check",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    if args.device == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            parser.error("set PYTORCH_ENABLE_MPS_FALLBACK=0")
        if not torch.backends.mps.is_available():
            parser.error("MPS unavailable; no automatic CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    collection = json.loads(args.collection.read_text())
    if collection["status"] != "complete":
        parser.error("collection must have completed successfully")
    remaining = collection["deadline_unix"] - time.time()
    if remaining <= 0:
        parser.error("the collection's total run budget has already expired")
    deadline = time.monotonic() + remaining
    parent = Path(collection["checkpoint"])
    if sha256(parent) != collection["checkpoint_sha256"]:
        parser.error("parent checkpoint no longer matches the collection")
    records = [g for g in collection["games"] if g["completed"]]
    if len(records) < collection["minimum_completed_games"]:
        parser.error("too few completed games")
    if len({g["path"] for g in records}) != len(records):
        parser.error("duplicate game files")
    train_ids, held_out_ids = split_game_indices(len(records), seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "loading",
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": args.device,
        "compute": "fp32" if args.fp32 else "bf16 autocast",
        "parameter_gradient_optimizer_loss_dtype": "float32",
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "parent_checkpoint": str(parent),
        "parent_sha256": collection["checkpoint_sha256"],
        "collection_manifest": str(args.collection.resolve()),
        "collection_sha256": sha256(args.collection),
        "deadline_unix": collection["deadline_unix"],
        "split_seed": args.seed,
        "shuffle_seed": args.seed,
        "train_games": [records[i] for i in train_ids],
        "held_out_games": [records[i] for i in held_out_ids],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": {
            "name": "AdamW",
            "fresh_state": True,
            "lr": args.learning_rate,
            "weight_decay": 0,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "foreach": False,
        },
        "value_weight": 1.0,
    }
    metrics = args.output_dir / "metrics.json"

    def save_report():
        temp = metrics.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(metrics)

    save_report()
    started = time.monotonic()
    try:
        games = []
        for entry in records:
            if time.monotonic() >= deadline:
                raise TimeoutError("run budget expired while loading data")
            path = Path(entry["path"])
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"game checksum changed: {path}")
            (game,) = read_games(path)
            if (
                len(game.examples) != entry["examples"]
                or game.adjudication != entry["outcome"]
                or game.winner != entry["winner"]
            ):
                raise ValueError(f"game disagrees with manifest: {path}")
            games.append(game)
        train = [e for i in train_ids for e in games[i].examples]
        held_out = [e for i in held_out_ids for e in games[i].examples]
        report.update(
            {
                "train_positions": len(train),
                "held_out_positions": len(held_out),
                "train_wdl_counts": [
                    sum(e.value_target[i] == 1 for e in train) for i in range(3)
                ],
                "held_out_wdl_counts": [
                    sum(e.value_target[i] == 1 for e in held_out) for i in range(3)
                ],
                "data_loading_seconds": time.monotonic() - started,
            }
        )
        loaded = load_checkpoint(parent, device=args.device)
        model = loaded.model
        report.update(
            {
                "parent_step": loaded.step,
                "model_config": asdict(model.config),
                "parameters": sum(p.numel() for p in model.parameters()),
                "status": "training",
            }
        )
        save_report()
        print(
            f"Loaded {len(train)} training / {len(held_out)} held-out positions; {len(train_ids)} / {len(held_out_ids)} whole games",
            flush=True,
        )
        train_start = time.monotonic()
        with (args.output_dir / "events.jsonl").open("x") as events:

            def event(row):
                row = {
                    **row,
                    "elapsed_training_seconds": time.monotonic() - train_start,
                }
                events.write(json.dumps(row) + "\n")
                events.flush()
                if row["kind"] == "evaluation" or row["step"] % 50 == 0:
                    print(json.dumps(row), flush=True)

            result = train_minibatches(
                model,
                train,
                held_out,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                bf16=not args.fp32,
                deadline=deadline,
                on_event=event,
            )
        if args.device == "mps":
            torch.mps.synchronize()
        report.update(
            {
                **asdict(result),
                "training_seconds_including_evaluations": time.monotonic()
                - train_start,
                "sample_reuse_ratio": result.samples_processed / len(train),
            }
        )
        if not all(change > 0 for change in result.parameter_max_changes.values()):
            raise RuntimeError("expected trunk and both heads to change")
        if time.monotonic() >= deadline:
            raise TimeoutError("run budget expired before checkpoint export")
        final_step = loaded.step + result.steps
        # Expose model.pt only after validating the saved snapshot.
        partial = args.output_dir / "model.pt.partial"
        save_checkpoint(partial, model, step=final_step)
        reloaded = load_checkpoint(partial, device=args.device)
        if reloaded.step != final_step or reloaded.model.config != model.config:
            raise RuntimeError("checkpoint metadata changed on reload")
        for name, tensor in model.state_dict().items():
            if not torch.equal(tensor.cpu(), reloaded.model.state_dict()[name].cpu()):
                raise RuntimeError(f"checkpoint state changed on reload: {name}")
        probe = collate_examples(held_out[:4]).features.to(args.device)
        with torch.no_grad():
            before = model.eval()(probe)
            after = reloaded.model(probe)
        errors = []
        for a, b in zip(before, after, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            errors.append((a - b).abs().max().item())
        if sha256(parent) != collection["checkpoint_sha256"]:
            raise RuntimeError("parent checkpoint changed during training")
        checkpoint = args.output_dir / "model.pt"
        partial.rename(checkpoint)
        report.update(
            {
                "status": "complete",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_step": final_step,
                "reload_fp32_max_absolute_errors": errors,
                "checkpoint_state_exact": True,
            }
        )
        print(f"Checkpoint B: {checkpoint.resolve()}", flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_unix"] = time.time()
        save_report()


if __name__ == "__main__":
    main()
