"""Overfit the tiny model on a fixed batch of real, completed-game records."""

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

from neurodiktyon import ChessModel, ModelConfig
from neurodiktyon.checkpoint import load_checkpoint, save_checkpoint
from neurodiktyon.training import train_fixed_batch
from neurodiktyon.training_data import (
    INPUT_ENCODING,
    POLICY_VOCABULARY,
    TrainingBatch,
    collate_examples,
    read_games,
)

TINY_CONFIG = ModelConfig(
    n_blocks=2,
    d_model=64,
    n_heads=4,
    d_ff=64,
    gab_d1=8,
    gab_d2=32,
    gab_d3=32,
    d_policy=64,
    d_value_hidden=32,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="JSONL from self_play_export")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.device == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            parser.error(
                "run with PYTORCH_ENABLE_MPS_FALLBACK=0 to rule out CPU fallback"
            )
        if not torch.backends.mps.is_available():
            parser.error(
                "MPS is unavailable in this process; no automatic CPU fallback"
            )
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    games = list(read_games(args.data))
    examples = [example for game in games for example in game.examples]
    cpu_batch = collate_examples(examples)
    labels = cpu_batch.value_targets.sum(dim=0).to(torch.int64).tolist()
    if min(labels) == 0:
        parser.error("this smoke test needs at least one win, draw, and loss example")
    if torch.unique(cpu_batch.features.flatten(1), dim=0).shape[0] != len(examples):
        parser.error("use distinct inputs for this fixed-batch memorization test")
    # Soft policy targets have an entropy floor: zero cross-entropy isn't the goal.
    positive = cpu_batch.policy_targets[cpu_batch.policy_targets > 0]
    entropy = -(positive * positive.log()).sum().item() / len(examples)
    batch = TrainingBatch(*(tensor.to(args.device) for tensor in cpu_batch))
    model = ChessModel(**asdict(TINY_CONFIG)).to(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"Fixed batch: {len(examples)} positions from {len(games)} completed games; "
        f"W/D/L counts={labels}.\n"
        f"Model: {sum(p.numel() for p in model.parameters()):,} parameters; "
        f"device={args.device}, FP32, AdamW lr={args.learning_rate:g}, "
        f"weight_decay=0, steps={args.steps}, seed={args.seed}.\n"
        f"Policy target entropy={entropy:.6f} nats/position.",
        flush=True,
    )
    if args.device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    result = train_fixed_batch(
        model, batch, steps=args.steps, learning_rate=args.learning_rate
    )
    if args.device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.inference_mode():
        before_reload = tuple(tensor.cpu() for tensor in model(batch.features))
    checkpoint = args.output_dir / "model.pt"
    save_checkpoint(checkpoint, model, step=args.steps)
    loaded = load_checkpoint(checkpoint, device=args.device)
    with torch.inference_mode():
        after_reload = tuple(tensor.cpu() for tensor in loaded.model(batch.features))
    differences = {}
    for name, before, after in zip(
        ("policy_logits", "value_logits"),
        before_reload,
        after_reload,
        strict=True,
    ):
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
        differences[name] = (before - after).abs().max().item()
    if loaded.step != args.steps or loaded.model.config != model.config:
        raise AssertionError("checkpoint metadata changed during reload")
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor.cpu(), loaded.model.state_dict()[name].cpu()):
            raise AssertionError(f"checkpoint state changed during reload: {name}")
    first, last = result.losses[0], result.losses[-1]
    checks = {
        "all_gradients_present_and_finite": result.gradient_checks == args.steps,
        "trunk_and_both_heads_changed": all(
            x > 0 for x in result.parameter_max_changes.values()
        ),
        "policy_loss_decreased": last.policy < first.policy,
        "value_loss_decreased": last.value < first.value,
        "checkpoint_state_exact": True,
        "checkpoint_predictions_close": True,
    }
    report = {
        "purpose": "fixed-batch training smoke test; not a strength or throughput benchmark",
        "cwd": str(Path.cwd()),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "threads": torch.get_num_threads(),
        "device": args.device,
        "dtype": "float32",
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "data": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "completed_games": len(games),
        "input_shape": list(batch.features.shape),
        "wdl_counts": labels,
        "input_encoding": INPUT_ENCODING,
        "policy_vocabulary": POLICY_VOCABULARY,
        "seed": args.seed,
        "model_config": asdict(model.config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer": {
            "name": "AdamW",
            "lr": args.learning_rate,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "foreach": False,
        },
        "value_weight": 1.0,
        "policy_target_entropy": entropy,
        "final_policy_kl": last.policy - entropy,
        "training_elapsed_with_checks_and_first_use_seconds": elapsed,
        **asdict(result),
        "checkpoint": str(checkpoint.resolve()),
        "reload_max_absolute_errors": differences,
        "checks": checks,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in result.losses:
        if row.step % 20 == 0 or row.step == args.steps:
            print(
                f"step={row.step:3d}  policy={row.policy:.6f}  value={row.value:.6f}  total={row.total:.6f}"
            )
    print(
        f"Final policy KL above target entropy: {last.policy - entropy:.6f} nats/position"
    )
    print(f"Maximum parameter changes: {result.parameter_max_changes}")
    print(f"Reload maximum absolute logit differences: {differences}")
    print(f"Checks: {checks}")
    print(f"Artifacts: {args.output_dir.resolve()}")
    if not all(checks.values()):
        raise RuntimeError("smoke-test checks failed; inspect metrics.json")


if __name__ == "__main__":
    main()
