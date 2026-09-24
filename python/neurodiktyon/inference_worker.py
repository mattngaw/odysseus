"""Persistent CPU inference from a checkpoint or random seed-0 weights.

Protocol v1: startup is <4sIII: ODNN, version, input float count, output float
count. Each request contains 64*110 little-endian float32 features in square-major
order; each reply contains 1858 policy logits followed by 3 W/D/L logits in the
same format. One request is in flight at a time. EOF between requests exits;
malformed/truncated messages fail the worker. Diagnostics use stderr only.
Startup failures may instead send <4sIII: ODNE, version=1, UTF-8 byte count,
reserved=0, followed by at most 4096 message bytes, then exit. This lets UCI
surface checkpoint-load errors. Successful request/reply framing is unchanged.

Version changes must cover encoding/vocabulary semantics as well as shapes.
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .embedding import FEATURE_COUNT, SQUARE_COUNT
from .model import ChessModel
from .policy_map import POLICY_SIZE

INPUT_FLOATS = SQUARE_COUNT * FEATURE_COUNT
OUTPUT_FLOATS = POLICY_SIZE + 3
HANDSHAKE = struct.pack("<4sIII", b"ODNN", 1, INPUT_FLOATS, OUTPUT_FLOATS)


def write_startup_error(destination: BinaryIO, message: str) -> None:
    """Emit a bounded UTF-8 failure frame before the normal ready handshake."""
    payload = (message or "unknown startup error").encode("utf-8", errors="replace")[
        :4096
    ]
    payload = payload.decode("utf-8", errors="ignore").encode("utf-8")
    destination.write(struct.pack("<4sIII", b"ODNE", 1, len(payload), 0))
    destination.write(payload)
    destination.flush()


def read_request(source: BinaryIO) -> bytes | None:
    """Read one complete request, allowing pipe reads to split anywhere."""
    payload = bytearray()
    while len(payload) < INPUT_FLOATS * 4:
        chunk = source.read(INPUT_FLOATS * 4 - len(payload))
        if not chunk:
            if not payload:
                return None
            raise EOFError("truncated inference request")
        payload.extend(chunk)
    return bytes(payload)


def serve(model: ChessModel, source: BinaryIO, destination: BinaryIO) -> None:
    model.eval()
    destination.write(HANDSHAKE)
    destination.flush()
    with torch.inference_mode():
        while (payload := read_request(source)) is not None:
            # Own a native-endian, writable array; the pipe's byte buffer is immutable.
            array = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
            features = torch.from_numpy(array).reshape(1, SQUARE_COUNT, FEATURE_COUNT)
            if not torch.isfinite(features).all():
                raise ValueError("inference features must be finite")
            output = model(features)
            if output.policy_logits.shape != (
                1,
                POLICY_SIZE,
            ) or output.value_logits.shape != (1, 3):
                raise ValueError("model returned incompatible policy/value shapes")
            logits = torch.cat((output.policy_logits, output.value_logits), dim=1)
            if not torch.isfinite(logits).all():
                raise ValueError("model returned nonfinite logits")
            destination.write(logits.cpu().float().numpy().astype("<f4").tobytes())
            destination.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="FP32 model snapshot; seed applies only without this option",
    )
    args = parser.parse_args()
    if not 0 <= args.seed < 2**64:
        parser.error("seed must be in [0, 2**64)")
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.checkpoint is None:
        torch.manual_seed(args.seed)
        model = ChessModel().cpu().float()
        description = f"untrained CPU FP32 model; seed={args.seed}"
    else:
        try:
            loaded = load_checkpoint(args.checkpoint, device="cpu")
        except Exception as error:
            # Never continue with random weights after an explicitly selected
            # checkpoint fails, including malformed torch serialization files.
            message = f"checkpoint {args.checkpoint}: {error}"
            write_startup_error(sys.stdout.buffer, message)
            raise RuntimeError(message) from error
        model = loaded.model
        description = (
            f"checkpoint CPU FP32 model; path={args.checkpoint}; step={loaded.step}; "
            f"blocks={model.config.n_blocks}; d_model={model.config.d_model}"
        )
    print(
        f"neurodiktyon: {description}; threads={args.threads}; "
        f"parameters={sum(p.numel() for p in model.parameters()):,}",
        file=sys.stderr,
        flush=True,
    )
    serve(model, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    try:
        main()
    except (OSError, EOFError, ValueError, RuntimeError) as error:
        print(f"neurodiktyon worker failed: {error}", file=sys.stderr)
        sys.exit(1)
