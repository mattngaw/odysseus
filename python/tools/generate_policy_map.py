"""Generate packaged policy indices from Pyxis's authoritative vocabulary dump."""

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "python/neurodiktyon/_policy_map_data.py"
BASE_COUNT = 1792
POLICY_SIZE = 1858


def generate() -> str:
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "pyxis",
            "--example",
            "policy_vocabulary",
            "--locked",
            "--offline",
            "--",
            "--dump",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    labels = result.stdout.splitlines()
    if len(labels) != POLICY_SIZE or len(set(labels)) != POLICY_SIZE:
        raise ValueError("expected 1,858 unique vocabulary labels from Pyxis")

    base_indices = []
    promotion_indices = []
    for index, label in enumerate(labels):
        match = re.fullmatch(r"([a-h])([1-8])([a-h])([1-8])([qrb]?)", label)
        if match is None:
            raise ValueError(f"invalid vocabulary label at {index}: {label!r}")
        from_file, from_rank, to_file, to_rank, piece = match.groups()
        source_file = ord(from_file) - ord("a")
        destination_file = ord(to_file) - ord("a")
        if index < BASE_COUNT:
            if piece:
                raise ValueError(f"expected a base entry at {index}: {label}")
            source = (int(from_rank) - 1) * 8 + source_file
            destination = (int(to_rank) - 1) * 8 + destination_file
            base_indices.append(source * 64 + destination)
        else:
            if (
                not piece
                or from_rank != "7"
                or to_rank != "8"
                or abs(source_file - destination_file) > 1
            ):
                raise ValueError(f"expected a promotion entry at {index}: {label}")
            promotion_indices.append(
                (source_file * 8 + destination_file) * 3 + "qrb".index(piece)
            )

    digest = hashlib.sha256(("\n".join(labels) + "\n").encode()).hexdigest()
    lines = [
        '"""Generated policy gather indices; do not edit by hand."""',
        "",
        "# Regenerate: python python/tools/generate_policy_map.py",
        "# Verify against Pyxis: python python/tools/generate_policy_map.py --check",
        "# Source: crates/pyxis/examples/policy_vocabulary.rs --dump",
        f"# SHA256 of newline-delimited vocabulary labels: {digest}",
        "",
        "# fmt: off",
    ]
    for name, indices in [
        ("BASE_INDICES", base_indices),
        ("PROMOTION_INDICES", promotion_indices),
    ]:
        lines.append(f"{name} = (")
        for start in range(0, len(indices), 12):
            lines.append(
                "    " + ", ".join(map(str, indices[start : start + 12])) + ","
            )
        lines.extend([")", ""])
    lines.append("# fmt: on")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args()
    expected = generate()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != expected:
            parser.exit(1, f"Stale or missing {OUTPUT}; rerun without --check.\n")
        print("All 1,858 packaged gather indices match the current Pyxis vocabulary.")
    else:
        OUTPUT.write_text(expected)
        print(f"Generated {OUTPUT.relative_to(PROJECT_ROOT)} from 1,858 Pyxis entries.")


if __name__ == "__main__":
    main()
