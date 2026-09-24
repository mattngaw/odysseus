"""Read versioned completed-game JSONL and assemble CPU training batches.

Each line is one whole completed game, with raw visits for every legal slot.
Validation checks the interchange contract and label perspectives; it does not
replay chess or prove that the producer really reached the stated outcome.
"""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor

from .embedding import FEATURE_COUNT, SQUARE_COUNT
from .policy_map import POLICY_SIZE

FORMAT = "odysseus.self_play"
VERSION = 1
INPUT_ENCODING = "pyxis-110-v1"
POLICY_VOCABULARY = "lc0-1858-v1"

_DRAW_KINDS = {
    "automatic_stalemate",
    "automatic_insufficient_material",
    "automatic_fivefold_repetition",
    "automatic_seventy_five_move_rule",
    "self_play_threefold_repetition",
}


@dataclass(frozen=True)
class TrainingExample:
    """A validated root; visits and their order are retained without rounding."""

    features: Tensor  # CPU FP32 [64,110], already player-relative.
    legal_indices: tuple[int, ...]
    visits: tuple[int, ...]
    total_visits: int
    value_target: tuple[float, float, float]  # W/D/L for side_to_move.
    ply: int  # Relative to the supplied starting position, not the FEN clock.
    side_to_move: str


@dataclass(frozen=True)
class CompletedGameRecord:
    adjudication: str
    winner: str | None
    examples: tuple[TrainingExample, ...]


class TrainingBatch(NamedTuple):
    features: Tensor  # FP32 [B,64,110]
    policy_targets: Tensor  # FP32 [B,1858]
    legal_mask: Tensor  # bool [B,1858]
    value_targets: Tensor  # FP32 [B,3]


def read_games(path: str | PathLike[str]) -> Iterator[CompletedGameRecord]:
    """Stream completed games, rejecting incompatible/malformed lines explicitly.

    Empty files yield no games; empty completed games yield no training examples.
    Blank lines are errors. A final complete JSON object needs no trailing newline,
    but an incomplete final object is an error, never silently skipped. Earlier
    games may have been yielded before a later line fails. Memory is bounded by
    the current game's size, not by the whole file.
    """
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            try:
                raw = json.loads(
                    line, object_pairs_hook=_unique_fields, parse_constant=_bad_constant
                )
                yield _parse_game(raw)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def collate_examples(examples: Sequence[TrainingExample]) -> TrainingBatch:
    """Batch examples from read_games; normalize raw visits without temperature.

    Outputs are CPU tensors. Device placement belongs to the training caller.
    Every supplied legal index is masked in, even when its visit count is zero.
    """
    if not examples:
        raise ValueError("cannot collate an empty batch")
    policy = torch.zeros(len(examples), POLICY_SIZE, dtype=torch.float32)
    mask = torch.zeros(len(examples), POLICY_SIZE, dtype=torch.bool)
    for row, example in enumerate(examples):
        indices = torch.tensor(example.legal_indices, dtype=torch.long)
        # Divide exact integer counts in double precision, then round to FP32,
        # matching RecordedRoot::policy_target in Rust.
        policy[row, indices] = torch.tensor(
            [count / example.total_visits for count in example.visits],
            dtype=torch.float32,
        )
        mask[row, indices] = True
    return TrainingBatch(
        torch.stack([example.features for example in examples]),
        policy,
        mask,
        torch.tensor([e.value_target for e in examples], dtype=torch.float32),
    )


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _fields(value, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or value.keys() != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")


def _integer(value, name: str, upper: int = (1 << 64) - 1) -> int:
    if type(value) is not int or not 0 <= value <= upper:
        raise ValueError(f"{name} must be an integer in [0,{upper}]")
    return value


def _color(value) -> str:
    if value not in ("white", "black"):
        raise ValueError("color must be white or black")
    return value


def _parse_game(raw) -> CompletedGameRecord:
    _fields(
        raw,
        {
            "format",
            "version",
            "input_encoding",
            "policy_vocabulary",
            "adjudication",
            "examples",
        },
        "game",
    )
    if (
        raw["format"] != FORMAT
        or type(raw["version"]) is not int
        or raw["version"] != VERSION
        or raw["input_encoding"] != INPUT_ENCODING
        or raw["policy_vocabulary"] != POLICY_VOCABULARY
    ):
        raise ValueError(
            "unsupported format, version, input encoding, or policy vocabulary"
        )
    outcome = raw["adjudication"]
    if not isinstance(outcome, dict) or type(outcome.get("kind")) is not str:
        raise ValueError("adjudication must have a recognized kind")
    kind = outcome["kind"]
    winner = None
    if kind == "automatic_checkmate":
        _fields(outcome, {"kind", "winner"}, "checkmate")
        winner = _color(outcome["winner"])
    elif kind in _DRAW_KINDS:
        _fields(outcome, {"kind"}, "draw")
    else:
        raise ValueError(f"unsupported adjudication: {kind}")
    if not isinstance(raw["examples"], list):
        raise ValueError("examples must be a list")
    examples = []
    for item in raw["examples"]:
        example = _parse_example(item, winner)
        if examples:
            previous = examples[-1]
            gap = example.ply - previous.ply
            same_side = example.side_to_move == previous.side_to_move
            if gap <= 0 or same_side != (gap % 2 == 0):
                raise ValueError(
                    "plies must increase with consistent side-to-move parity"
                )
        examples.append(example)
    return CompletedGameRecord(kind, winner, tuple(examples))


def _parse_example(raw, winner: str | None) -> TrainingExample:
    _fields(
        raw,
        {"ply", "side_to_move", "input", "policy", "total_visits", "value_target"},
        "example",
    )
    ply = _integer(raw["ply"], "ply")
    side = _color(raw["side_to_move"])
    features = raw["input"]
    if (
        not isinstance(features, list)
        or len(features) != SQUARE_COUNT
        or any(
            not isinstance(row, list) or len(row) != FEATURE_COUNT for row in features
        )
    ):
        raise ValueError("input must have shape [64,110]")
    for row in features:
        for channel, value in enumerate(row):
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError("input features must be finite numbers in [0,1]")
            if channel < FEATURE_COUNT - 1 and value not in (0, 1):
                raise ValueError(
                    "input features except the halfmove clock must be binary"
                )
    policy = raw["policy"]
    if not isinstance(policy, list) or not 1 <= len(policy) <= POLICY_SIZE:
        raise ValueError("policy must contain at least one legal slot")
    indices, visits = [], []
    for entry in policy:
        _fields(entry, {"index", "visits"}, "policy entry")
        indices.append(_integer(entry["index"], "policy index", POLICY_SIZE - 1))
        visits.append(_integer(entry["visits"], "visits", (1 << 32) - 1))
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate legal policy index")
    total = _integer(raw["total_visits"], "total_visits")
    if total == 0 or total != sum(visits):
        raise ValueError("total_visits must equal a positive sum of raw visits")
    expected = (
        (0.0, 1.0, 0.0)
        if winner is None
        else ((1.0, 0.0, 0.0) if side == winner else (0.0, 0.0, 1.0))
    )
    value = raw["value_target"]
    if (
        not isinstance(value, list)
        or any(type(x) not in (int, float) for x in value)
        or tuple(value) != expected
    ):
        raise ValueError("W/D/L label disagrees with adjudication and side to move")
    return TrainingExample(
        torch.tensor(features, dtype=torch.float32),
        tuple(indices),
        tuple(visits),
        total,
        expected,
        ply,
        side,
    )
