import copy
import json

import pytest
import torch

from neurodiktyon.training_data import collate_examples, read_games


@pytest.fixture
def record():
    # Structural fixture, not a claim that these zero planes describe legal chess.
    return {
        "format": "odysseus.self_play",
        "version": 1,
        "input_encoding": "pyxis-110-v1",
        "policy_vocabulary": "lc0-1858-v1",
        "adjudication": {"kind": "automatic_checkmate", "winner": "black"},
        "examples": [
            {
                "ply": 0,
                "side_to_move": "white",
                "input": [[0.0] * 110 for _ in range(64)],
                "policy": [
                    {"index": 322, "visits": 60},
                    {"index": 293, "visits": 30},
                    {"index": 159, "visits": 10},
                    {"index": 97, "visits": 0},
                ],
                "total_visits": 100,
                "value_target": [0.0, 0.0, 1.0],
            }
        ],
    }


def load(tmp_path, *records):
    path = tmp_path / "games.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return list(read_games(path))


def test_batch_keeps_raw_counts_legal_zeros_and_each_players_outcome(record, tmp_path):
    white = record["examples"][0]
    white["input"][45][7] = 1
    white["input"][0][109] = 1 / 150
    black = copy.deepcopy(white)
    black.update(ply=3, side_to_move="black", value_target=[1, 0, 0])
    # Gaps in recording are supported; the side changes with odd ply gaps.
    record["examples"].append(black)
    (game,) = load(tmp_path, record)
    assert game.adjudication == "automatic_checkmate"
    assert game.winner == "black"
    assert game.examples[0].legal_indices == (322, 293, 159, 97)
    assert game.examples[0].visits == (60, 30, 10, 0)
    assert game.examples[0].total_visits == 100
    batch = collate_examples(game.examples)
    assert batch.features.shape == (2, 64, 110)
    assert batch.policy_targets.shape == batch.legal_mask.shape == (2, 1858)
    assert batch.value_targets.shape == (2, 3)
    assert batch.legal_mask.dtype == torch.bool
    for tensor in (batch.features, batch.policy_targets, batch.value_targets):
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
    assert batch.features[0, 45, 7].item() == 1
    assert batch.features[0, 0, 109].item() == torch.tensor(1 / 150).item()
    torch.testing.assert_close(
        batch.policy_targets[0, [322, 293, 159, 97]],
        torch.tensor([0.6, 0.3, 0.1, 0.0]),
        rtol=0,
        atol=0,
    )
    assert batch.legal_mask[0, 97] and batch.policy_targets[0, 97] == 0
    assert not batch.legal_mask[0, 98] and batch.policy_targets[0, 98] == 0
    assert batch.legal_mask.sum().item() == 8
    assert batch.policy_targets.sum().item() == 2
    assert batch.value_targets.tolist() == [[0, 0, 1], [1, 0, 0]]


def test_large_raw_integer_visits_survive_before_normalization(record, tmp_path):
    example = record["examples"][0]
    counts = [(1 << 32) - 1, (1 << 24) + 1, 1, 0]
    for entry, count in zip(example["policy"], counts, strict=True):
        entry["visits"] = count
    example["total_visits"] = sum(counts)
    (game,) = load(tmp_path, record)
    assert game.examples[0].visits == tuple(counts)
    assert game.examples[0].total_visits == sum(counts)
    batch = collate_examples(game.examples)
    expected = torch.tensor([n / sum(counts) for n in counts])
    torch.testing.assert_close(
        batch.policy_targets[0, [322, 293, 159, 97]], expected, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "kind",
    [
        "automatic_stalemate",
        "automatic_insufficient_material",
        "automatic_fivefold_repetition",
        "automatic_seventy_five_move_rule",
        "self_play_threefold_repetition",
    ],
)
def test_draw_reasons_remain_distinct(record, tmp_path, kind):
    record["adjudication"] = {"kind": kind}
    record["examples"][0]["value_target"] = [0, 1, 0]
    (game,) = load(tmp_path, record)
    assert game.adjudication == kind
    assert game.winner is None
    assert collate_examples(game.examples).value_targets.tolist() == [[0, 1, 0]]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("format",), "unknown", "unsupported"),
        (("version",), 2, "unsupported"),
        (("version",), True, "unsupported"),
        (("input_encoding",), "pyxis-110-v2", "unsupported"),
        (("policy_vocabulary",), "wrong-order", "unsupported"),
        (("adjudication",), {"kind": "truncated"}, "unsupported adjudication"),
        (("adjudication", "winner"), "red", "color"),
        (("examples",), None, "list"),
        (("examples", 0, "side_to_move"), "red", "color"),
        (("examples", 0, "ply"), -1, "ply"),
        (("examples", 0, "input"), [[0] * 110] * 63, "shape"),
        (("examples", 0, "input", 0), [0] * 109, "shape"),
        (("examples", 0, "input", 0, 0), 0.5, "binary"),
        (("examples", 0, "input", 0, 0), True, "finite numbers"),
        (("examples", 0, "input", 0, 109), 1.1, "finite numbers"),
        (("examples", 0, "input", 0, 109), float("nan"), "non-finite"),
        (("examples", 0, "input", 0, 109), float("inf"), "non-finite"),
        (("examples", 0, "policy"), [], "legal slot"),
        (("examples", 0, "policy", 0, "index"), 293, "duplicate"),
        (("examples", 0, "policy", 0, "index"), 1858, "policy index"),
        (("examples", 0, "policy", 0, "index"), -1, "policy index"),
        (("examples", 0, "policy", 0, "visits"), -1, "visits"),
        (("examples", 0, "policy", 0, "visits"), 1 << 32, "visits"),
        (("examples", 0, "policy", 0, "visits"), 60.0, "visits"),
        (("examples", 0, "policy", 0, "visits"), True, "visits"),
        (("examples", 0, "total_visits"), 101, "sum"),
        (("examples", 0, "total_visits"), 0, "positive"),
        (("examples", 0, "value_target"), [1, 0, 0], "W/D/L"),
        (("examples", 0, "value_target"), [0, 0, True], "W/D/L"),
        (("examples", 0, "value_target"), [0.5, 0, 0.5], "W/D/L"),
    ],
)
def test_bad_records_fail_with_line_number(record, tmp_path, path, value, message):
    parent = record
    for field in path[:-1]:
        parent = parent[field]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match=rf"games.jsonl:1: .*{message}"):
        load(tmp_path, record)


def test_rejects_zero_visit_policy_and_out_of_order_roots(record, tmp_path):
    for entry in record["examples"][0]["policy"]:
        entry["visits"] = 0
    record["examples"][0]["total_visits"] = 0
    with pytest.raises(ValueError, match="positive"):
        load(tmp_path, record)
    record["examples"][0]["policy"][0]["visits"] = 1
    record["examples"][0]["total_visits"] = 1
    second = copy.deepcopy(record["examples"][0])
    record["examples"].append(second)
    with pytest.raises(ValueError, match="plies must increase"):
        load(tmp_path, record)
    second["ply"] = 1  # Wrong color parity.
    with pytest.raises(ValueError, match="parity"):
        load(tmp_path, record)
    second["ply"] = 2
    assert len(load(tmp_path, record)[0].examples) == 2


def test_stream_boundaries_empty_games_and_incomplete_tail(record, tmp_path):
    path = tmp_path / "games.jsonl"
    path.write_text("")
    assert list(read_games(path)) == []
    empty = copy.deepcopy(record)
    empty["examples"] = []
    games = load(tmp_path, record, empty, record)
    assert [len(game.examples) for game in games] == [1, 0, 1]
    with pytest.raises(ValueError, match="empty batch"):
        collate_examples(games[1].examples)
    path.write_text(json.dumps(record))  # Complete final object, no newline.
    assert len(list(read_games(path))) == 1
    path.write_text(json.dumps(record) + '\n{"version":')
    stream = read_games(path)
    assert len(next(stream).examples) == 1
    with pytest.raises(ValueError, match="games.jsonl:2:"):
        next(stream)


def test_duplicate_missing_and_extra_json_fields_fail(record, tmp_path):
    path = tmp_path / "games.jsonl"
    path.write_text(
        json.dumps(record).replace('"version": 1', '"version": 1, "version": 1')
    )
    with pytest.raises(ValueError, match="duplicate JSON field"):
        list(read_games(path))
    record["examples"][0]["unknown"] = 1
    with pytest.raises(ValueError, match="exactly"):
        load(tmp_path, record)
    del record["examples"][0]["unknown"]
    del record["examples"][0]["total_visits"]
    with pytest.raises(ValueError, match="exactly"):
        load(tmp_path, record)
    path.write_text("\n")
    with pytest.raises(ValueError, match="games.jsonl:1:"):
        list(read_games(path))
