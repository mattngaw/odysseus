"""Measure Cargo builds of the frozen review snapshots, using fresh targets."""

import argparse
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("before", "after"))
parser.add_argument("--label", help="Unique output/target prefix for another measurement")
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()
if args.runs < 1:
    parser.error("--runs must be positive")
label = args.label or args.stage
if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in label):
    parser.error("--label must use only ASCII letters, digits, hyphens, and underscores")

output = Path(__file__).resolve().parent
root = output.parents[2]
manifest = root / "target/reviews/magic-build-script" / args.stage / "Cargo.toml"
if not manifest.is_file():
    parser.error(f"missing source snapshot: {manifest}")
report = output / f"{label}.json"
if report.exists():
    parser.error(f"measurement already exists: {report}; use a new --label")

records = []
for trial in range(1, args.runs + 1):
    for profile in ("debug", "release"):
        name = f"{label}-{profile}-{trial}"
        target = root / "target/build-benchmarks/magic-build-script" / name
        if target.exists():
            parser.error(f"target must be absent for a clean build: {target}")
        command = ["cargo", "build", "--manifest-path", str(manifest), "-p", "penteconter",
                   "--lib", "--locked", "--offline", "--target-dir", str(target)]
        if profile == "release":
            command.append("--release")
        print(f"Starting {name}", flush=True)
        log = output / f"{name}.log"
        with log.open("x") as stream:
            started = time.perf_counter()
            result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - started
        records.append({"stage": args.stage, "profile": profile, "trial": trial,
                        "wall_seconds": elapsed, "exit_code": result.returncode,
                        "working_directory": str(root), "command": command, "log": str(log)})
        report.write_text(json.dumps(records, indent=2) + "\n")
        print(f"{name}: {elapsed:.3f} s, exit {result.returncode}", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
