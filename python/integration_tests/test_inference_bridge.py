"""Requires: cargo build -p odysseus --example inference_bridge --locked --offline."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from neurodiktyon import ChessModel
from neurodiktyon.checkpoint import save_checkpoint


@pytest.mark.parametrize("checkpoint", [False, True])
def test_rust_round_trip_matches_direct_model_for_real_encodings(tmp_path, checkpoint):
    root = Path(__file__).resolve().parents[2]
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
    executable = target / "debug" / "examples" / "inference_bridge"
    if sys.platform == "win32":
        executable = executable.with_suffix(".exe")
    dump = tmp_path / "round_trip.f32le"
    torch.manual_seed(19 if checkpoint else 0)
    model = (
        ChessModel(
            n_blocks=2, d_model=32, n_heads=4, d_ff=32, d_policy=16, d_value_hidden=16
        )
        if checkpoint
        else ChessModel()
    ).eval()
    command = [str(executable), sys.executable, str(dump)]
    if checkpoint:
        path = tmp_path / "Small  Network.pt"
        save_checkpoint(path, model, step=7)
        command.append(str(path))
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    assert "Repeated start-position request exactly matches" in result.stdout
    if checkpoint:
        assert "checkpoint CPU FP32 model;" in result.stderr
        assert "step=7; blocks=2; d_model=32;" in result.stderr
        assert "untrained" not in result.stderr
    else:
        assert "untrained CPU FP32 model; seed=0" in result.stderr
    records = np.fromfile(dump, dtype="<f4").reshape(6, 7040 + 1858 + 3)
    features = records[:, :7040].reshape(6, 64, 110).copy()
    logits = records[:, 7040:]
    np.testing.assert_array_equal(features[0], features[-1])
    np.testing.assert_array_equal(logits[0], logits[-1])
    # Real encoder coverage: Black's relative history, repetition, legal EP, clock.
    assert features[1, 45, 7] == 1  # White Nf3 -> relative f6, "their knight".
    assert features[1, 62, 13 + 7] == 1  # Previous white Ng1 -> relative g8.
    assert features[2, :, 12].tolist() == [1.0] * 64
    assert features[3, 43, 108] == 1  # Absolute d3 -> relative d6.
    assert features[3, 0, 104:108].tolist() == [0, 1, 1, 0]
    np.testing.assert_allclose(features[2, :, 109], np.float32(4 / 150))

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.inference_mode():
            for index in range(6):
                output = model(torch.from_numpy(features[index : index + 1]))
                expected = torch.cat((output.policy_logits, output.value_logits), dim=1)
                np.testing.assert_array_equal(logits[index], expected[0].numpy())
    finally:
        torch.set_num_threads(previous_threads)
