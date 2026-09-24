"""Requires: cargo build -p odysseus --bins --examples --locked --offline."""

import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from neurodiktyon import ChessModel
from neurodiktyon._policy_map_data import BASE_INDICES
from neurodiktyon.checkpoint import save_checkpoint

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
STATS = re.compile(r"info string (\S+) N: (\d+) \(P: ([\d.]+)%\) \(Q: ([+\-\d.]+)\)")


class Engine:
    def __init__(self, cwd, python=sys.executable, checkpoint=None, executable=None):
        self.process = subprocess.Popen(
            [str(executable or TARGET / "debug" / "odysseus")],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.stdout, self.stderr = queue.Queue(), queue.Queue()
        self.errors = []
        self.readers = []
        for stream, channel in [
            (self.process.stdout, self.stdout),
            (self.process.stderr, self.stderr),
        ]:
            reader = threading.Thread(
                target=self.read, args=(stream, channel), daemon=True
            )
            reader.start()
            self.readers.append(reader)
        self.send(f"setoption name NeuralPython value {python}")
        self.send("setoption name Evaluator value neural")
        if checkpoint is not None:
            self.send(f"setoption name NeuralCheckpoint value {checkpoint}")

    def read(self, stream, channel):
        for line in stream:
            if channel is self.stderr:
                self.errors.append(line.strip())
            channel.put(line.strip())
        channel.put(None)

    def send(self, text):
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()

    def until(self, prefix, *, stderr=False, timeout=15):
        channel = self.stderr if stderr else self.stdout
        deadline = time.monotonic() + timeout
        lines = []
        while True:
            line = channel.get(timeout=max(0, deadline - time.monotonic()))
            assert line is not None, (prefix, lines, self.errors)
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def search(self, command):
        self.send(command)
        lines = self.until("bestmove ")
        assert not any("failed:" in line or "rejected:" in line for line in lines), (
            lines
        )
        return lines

    def close(self):
        try:
            if self.process.poll() is None:
                self.send("quit")
                self.process.wait(timeout=3)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            for reader in self.readers:
                reader.join(timeout=3)
            for stream in (
                self.process.stdin,
                self.process.stdout,
                self.process.stderr,
            ):
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def stats(lines):
    result = {}
    for line in lines:
        if match := STATS.fullmatch(line):
            move, visits, prior, value = match.groups()
            result[move] = (int(visits), float(prior) / 100, float(value))
    return result


def policy_index(move, black):
    def square(text):
        index = (int(text[1]) - 1) * 8 + ord(text[0]) - ord("a")
        return index ^ 56 if black else index

    return BASE_INDICES.index(square(move[:2]) * 64 + square(move[2:4]))


@pytest.fixture
def saved_model(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        model = ChessModel(
            n_blocks=2, d_model=32, n_heads=4, d_ff=32, d_policy=16, d_value_hidden=16
        ).eval()
    path = tmp_path / "Small  Network.pt"
    save_checkpoint(path, model, step=7)
    return model, path


@pytest.mark.parametrize("checkpoint", [False, True])
def test_real_model_drives_priors_values_and_self_play_without_reloading(
    tmp_path, saved_model, checkpoint
):
    dump = tmp_path / "encodings.f32le"
    command = [
        str(TARGET / "debug" / "examples" / "inference_bridge"),
        sys.executable,
        str(dump),
    ]
    if checkpoint:
        command.append(str(saved_model[1]))
    subprocess.run(
        command,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=30,
    )
    records = np.fromfile(dump, dtype="<f4").reshape(6, 8901)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(0)
        model = saved_model[0] if checkpoint else ChessModel().eval()
        with torch.inference_mode():
            expected = [
                model(torch.from_numpy(row[:7040].copy().reshape(1, 64, 110)))
                for row in records[:2]
            ]
    finally:
        torch.set_num_threads(previous_threads)

    with Engine(tmp_path, checkpoint=saved_model[1] if checkpoint else None) as engine:
        engine.send("uci")
        handshake = engine.until("uciok")
        assert any("var neural" in line for line in handshake)
        assert "option name NeuralCheckpoint type string default <empty>" in handshake
        # Exercise both perspectives; only legal logits participate in the priors.
        for black, position in [(False, "startpos"), (True, "startpos moves g1f3")]:
            engine.send(f"position {position}")
            actual = stats(engine.search("go nodes 0"))
            assert len(actual) == 20
            indices = [policy_index(move, black) for move in actual]
            probabilities = (
                expected[int(black)].policy_logits[0, indices].double().softmax(dim=0)
            )
            np.testing.assert_allclose(
                [v[1] for v in actual.values()],
                probabilities.numpy(),
                rtol=0,
                atol=1e-8,
            )
            assert (
                max(v[1] for v in actual.values()) - min(v[1] for v in actual.values())
                > 0.001
            )

        engine.send("position startpos")
        one = stats(engine.search("go nodes 1 searchmoves g1f3"))
        wdl = expected[1].value_logits[0].double().softmax(dim=0)
        assert one["g1f3"][0] == 1
        assert abs(one["g1f3"][2] + float(wdl[0] - wdl[2])) < 1e-7
        # Replaying each chosen move validates it and retains the real game history.
        moves = []
        for _ in range(8):
            engine.send(
                "position startpos" + (" moves " + " ".join(moves) if moves else "")
            )
            lines = engine.search("go nodes 8")
            move = lines[-1].split()[1]
            assert move in stats(lines)
            moves.append(move)
        engine.send("position startpos moves " + " ".join(moves))
        engine.send("isready")
        assert engine.until("readyok") == ["readyok"]
        marker = (
            "checkpoint CPU FP32 model;"
            if checkpoint
            else "untrained CPU FP32 model; seed=0"
        )
        assert sum(marker in line for line in engine.errors) == 1
        if checkpoint:
            engine.send("position startpos\ngo infinite")
            engine.until("info time ")
            stopped = engine.search("stop")
            assert stopped[-1].split()[1] in stats(stopped)
            assert engine.search("go nodes 1")[-1] != "bestmove 0000"


def test_checkpoint_replacement_during_search_same_path_reload_and_clear(
    tmp_path, saved_model
):
    original, first_path = saved_model
    replacement = ChessModel(**vars(original.config)).eval()
    with torch.no_grad():
        replacement.policy_head.pairs.from_proj.weight.zero_()
        replacement.policy_head.pairs.from_proj.bias.zero_()
    second_path = tmp_path / "Replacement  Weights.pt"
    save_checkpoint(second_path, replacement, step=8)
    with Engine(tmp_path, checkpoint=first_path) as engine:
        engine.send("position startpos")
        original_stats = stats(engine.search("go nodes 0"))
        engine.send("go infinite")
        engine.until("info time ")
        engine.send(f"setoption name NeuralCheckpoint value {second_path}")
        engine.until("bestmove ")  # Old search ends before new settings take effect.
        uniform = stats(engine.search("go nodes 0"))
        assert all(abs(row[1] - 0.05) < 1e-8 for row in uniform.values())
        assert any(
            abs(original_stats[mv][1] - row[1]) > 0.001 for mv, row in uniform.items()
        )
        # Overwriting a file does not mutate an already loaded model.
        updated = tmp_path / "updated.pt"
        save_checkpoint(updated, original, step=9)
        updated.replace(second_path)
        engine.send("ucinewgame\nposition startpos")
        assert stats(engine.search("go nodes 0")) == uniform
        # Explicitly setting even the same path discards the cached process.
        engine.send(f"setoption name NeuralCheckpoint value {second_path}")
        assert stats(engine.search("go nodes 0")) == original_stats
        assert sum("checkpoint CPU FP32 model;" in line for line in engine.errors) == 3
        assert any("step=9;" in line for line in engine.errors)
        engine.send("setoption name NeuralCheckpoint value <empty>")
        cleared = engine.search("go nodes 0")
        assert any("untrained seed-0 CPU FP32 model" in line for line in cleared)
        assert (
            sum("untrained CPU FP32 model; seed=0" in line for line in engine.errors)
            == 1
        )


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("missing", "No such file"),
        ("incompatible", "unsupported checkpoint version, architecture, or encoding"),
        ("malformed", "checkpoint"),
    ],
)
def test_checkpoint_load_errors_reach_uci_and_valid_options_recover(
    tmp_path, saved_model, damage, reason
):
    bad_path = tmp_path / "Broken Checkpoint.pt"
    if damage == "incompatible":
        payload = torch.load(saved_model[1], weights_only=True)
        payload["input_encoding"] = "wrong-planes"
        torch.save(payload, bad_path)
    elif damage == "malformed":
        bad_path.write_bytes(b"not a torch checkpoint")
    with Engine(tmp_path, checkpoint=bad_path) as engine:
        engine.send("position fen k7/1Q6/2K5/8/8/8/8/8 b - - 0 1")
        assert (
            engine.search("go nodes 1")[-1] == "bestmove 0000"
        )  # No evaluation needed.
        engine.send("position startpos\ngo nodes 1")
        failure = engine.until("bestmove ")
        assert any(
            "worker startup failed:" in line and reason in line for line in failure
        )
        assert failure[-1] == "bestmove 0000"
        assert not any("untrained" in line for line in engine.errors)
        engine.send(f"setoption name NeuralCheckpoint value {saved_model[1]}")
        assert engine.search("go nodes 1")[-1] != "bestmove 0000"


def fake_worker(tmp_path, phase):
    # Use the configurable interpreter path to substitute a controlled wire peer.
    # Mixed case and spaces ensure setoption preserves path spelling.
    path = tmp_path / "Fake Neural Python"
    script = f"""#!{sys.executable}
import os, struct, sys, time
phase = {phase!r}
def pause():
    print('waiting ' + str(os.getpid()), file=sys.stderr, flush=True)
    time.sleep(60)
if phase == 'startup': pause()
sys.stdout.buffer.write(struct.pack('<4sIII', b'ODNN', 1, 7040, 1861))
sys.stdout.buffer.flush()
requests = 0
while sys.stdin.buffer.read(7040 * 4):
    requests += 1
    if phase == 'root' or (phase == 'leaf' and requests == 2): pause()
    if phase == 'bad':
        sys.stdout.buffer.write(struct.pack('<1861f', *([float('nan')] * 1861)))
    else:
        sys.stdout.buffer.write(bytes(1861 * 4))
    sys.stdout.buffer.flush()
"""
    path.write_text(script)
    path.chmod(0o755)
    return path


@pytest.mark.skipif(
    os.name != "posix", reason="executable Python fixture uses a shebang"
)
@pytest.mark.parametrize("phase", ["startup", "root", "leaf"])
def test_stop_interrupts_neural_work_and_returns_an_allowed_move(tmp_path, phase):
    with Engine(tmp_path, fake_worker(tmp_path, phase)) as engine:
        engine.send("position startpos")
        engine.send("go infinite searchmoves g1f3")
        marker = engine.until("waiting ", stderr=True)[-1]
        pid = int(marker.split()[1])
        engine.send("isready")
        engine.until("readyok", timeout=2)
        stopped = engine.search("stop")
        assert stopped[-1] == "bestmove g1f3"
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        # A queued replacement uses the next evaluator and cannot consume stale logits.
        engine.send("setoption name Evaluator value material")
        engine.send("position startpos")
        assert engine.search("go nodes 1")[-1] != "bestmove 0000"


@pytest.mark.skipif(
    os.name != "posix", reason="executable Python fixture uses a shebang"
)
@pytest.mark.parametrize("end", ["quit", "eof"])
def test_quit_or_eof_reaps_a_stalled_neural_worker(tmp_path, end):
    with Engine(tmp_path, fake_worker(tmp_path, "root")) as engine:
        engine.send("position startpos\ngo infinite")
        pid = int(engine.until("waiting ", stderr=True)[-1].split()[1])
        if end == "quit":
            engine.send("quit")
        else:
            engine.process.stdin.close()
        engine.process.wait(timeout=2)
        assert engine.process.returncode == 0
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_neural_failures_are_reported_and_terminal_roots_bypass_python(tmp_path):
    with Engine(tmp_path, tmp_path / "missing-python") as engine:
        engine.send("position fen k7/1Q6/2K5/8/8/8/8/8 b - - 0 1")
        assert engine.search("go nodes 8")[-1] == "bestmove 0000"
        engine.send("position startpos\ngo nodes 1")
        failure = engine.until("bestmove ")
        assert any("search failed: evaluation failed:" in line for line in failure)
        engine.send(f"setoption name NeuralPython value {sys.executable}")
        assert engine.search("go nodes 1")[-1] != "bestmove 0000"


@pytest.mark.skipif(
    os.name != "posix", reason="executable Python fixture uses a shebang"
)
def test_nonfinite_neural_reply_fails_explicitly_and_options_can_recover(tmp_path):
    with Engine(tmp_path, fake_worker(tmp_path, "bad")) as engine:
        engine.send("position startpos\ngo nodes 1")
        failure = engine.until("bestmove ")
        assert any("nonfinite logits" in line for line in failure)
        assert failure[-1] == "bestmove 0000"
        engine.send("setoption name Evaluator value uniform")
        assert engine.search("go nodes 1")[-1] != "bestmove 0000"
