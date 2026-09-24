import copy
import time
from collections import Counter

import pytest
import torch

from neurodiktyon import ChessModel
from neurodiktyon.minibatch_training import (
    evaluate_examples,
    split_game_indices,
    train_minibatches,
)
from neurodiktyon.training_data import TrainingExample


@pytest.fixture
def model_and_records():
    torch.manual_seed(3)
    model = ChessModel(
        n_blocks=1,
        d_model=8,
        n_heads=2,
        d_ff=12,
        gab_d1=2,
        gab_d2=4,
        gab_d3=3,
        d_policy=8,
        d_value_hidden=8,
    )
    records = []
    for i in range(7):
        features = torch.randn(64, 110)
        features[0, 0] = i  # Synthetic marker to audit exposure, not an encoder test.
        records.append(
            TrainingExample(
                features,
                (0, 1, 2, 3),
                (3, 2, 1, 0),
                6,
                tuple(float(j == i % 3) for j in range(3)),
                i,
                "white" if i % 2 == 0 else "black",
            )
        )
    return model, records[:5], records[5:]


def test_split_is_seeded_disjoint_and_contains_every_game():
    train, held = split_game_indices(20, seed=7)
    assert len(train) == 18 and len(held) == 2
    assert not set(train) & set(held)
    assert sorted(train + held) == list(range(20))
    assert (train, held) == split_game_indices(20, seed=7)
    assert (train, held) != split_game_indices(20, seed=8)
    assert tuple(map(len, split_game_indices(2, seed=0))) == (1, 1)
    with pytest.raises(ValueError, match="at least two"):
        split_game_indices(1, seed=0)


@pytest.mark.parametrize("bf16", [False, True])
def test_each_record_used_twice_no_held_out_updates_and_fp32_checkpoint_weights(
    model_and_records, bf16
):
    model, train, held = model_and_records
    exposure, events = [], []
    original = [e.features.clone() for e in train + held]

    def observe(module, inputs):
        if module.training and torch.is_grad_enabled():
            exposure.extend(inputs[0][:, 0, 0].tolist())

    hook = model.register_forward_pre_hook(observe)
    try:
        result = train_minibatches(
            model,
            train,
            held,
            batch_size=3,
            epochs=2,
            seed=19,
            bf16=bf16,
            on_event=events.append,
        )
    finally:
        hook.remove()
    assert Counter(exposure) == Counter({float(i): 2 for i in range(5)})
    assert result.steps == result.gradient_checks == 4
    assert result.samples_processed == 10
    assert [e["batch_size"] for e in events if e["kind"] == "update"] == [3, 2, 3, 2]
    assert [(e["epoch"], e["split"]) for e in result.evaluations] == [
        (0, "train"),
        (0, "held_out"),
        (1, "train"),
        (1, "held_out"),
        (2, "train"),
        (2, "held_out"),
    ]
    assert all(x > 0 for x in result.parameter_max_changes.values())
    assert all(p.dtype == p.grad.dtype == torch.float32 for p in model.parameters())
    assert all(
        torch.equal(before, e.features)
        for before, e in zip(original, train + held, strict=True)
    )


def test_seeded_training_replays_and_evaluation_weights_partial_batch_correctly(
    model_and_records,
):
    model, train, held = model_and_records
    other = copy.deepcopy(model)
    result = train_minibatches(model, train, held, batch_size=3, seed=9, bf16=False)
    replay = train_minibatches(other, train, held, batch_size=3, seed=9, bf16=False)
    assert result == replay
    assert all(
        torch.equal(p, q)
        for p, q in zip(model.parameters(), other.parameters(), strict=True)
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    by_batch = evaluate_examples(model, train, batch_size=3, bf16=False)
    single = [evaluate_examples(model, [e], batch_size=1, bf16=False) for e in train]
    for field in ("policy", "value", "total"):
        assert by_batch[field] == pytest.approx(
            sum(e[field] for e in single) / 5, abs=1e-6
        )
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())


@pytest.mark.parametrize(
    "options",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": float("nan")},
        {"deadline": time.monotonic() - 1},
    ],
)
def test_invalid_settings_or_expired_budget_do_not_mutate_weights(
    model_and_records, options
):
    model, train, held = model_and_records
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with pytest.raises((ValueError, TimeoutError)):
        train_minibatches(model, train, held, **options)
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())


def test_nonfinite_gradient_stops_before_update(model_and_records):
    model, train, held = model_and_records
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    hook = model.value_head.out_proj.weight.register_hook(
        lambda grad: torch.full_like(grad, torch.nan)
    )
    try:
        with pytest.raises(FloatingPointError, match="non-finite gradients"):
            train_minibatches(model, train, held, batch_size=3, bf16=False)
    finally:
        hook.remove()
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())
