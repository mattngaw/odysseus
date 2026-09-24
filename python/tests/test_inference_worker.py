import io
import struct

import numpy as np
import pytest
import torch

from neurodiktyon import ChessModel, ModelOutput
from neurodiktyon.inference_worker import (
    HANDSHAKE,
    INPUT_FLOATS,
    OUTPUT_FLOATS,
    serve,
    write_startup_error,
)


class FragmentedInput(io.BytesIO):
    def read(self, size=-1):
        # Split floats and requests across separate reads.
        return super().read(min(size, 31))


def small_model():
    return ChessModel(
        n_blocks=1,
        d_model=12,
        n_heads=3,
        d_ff=19,
        gab_d1=3,
        gab_d2=7,
        gab_d3=5,
        d_policy=7,
        d_value_hidden=9,
    )


def test_multiple_fragmented_requests_match_direct_inference_and_leave_weights_fixed():
    torch.manual_seed(0)
    model = small_model()
    features = torch.randn(2, 64, 110)
    requests = features.numpy().astype("<f4").tobytes()
    destination = io.BytesIO()
    weights_before = {name: p.detach().clone() for name, p in model.named_parameters()}
    serve(model, FragmentedInput(requests), destination)
    assert not model.training
    assert HANDSHAKE == struct.pack("<4sIII", b"ODNN", 1, 7040, 1861)
    wire = destination.getvalue()
    assert wire[:16] == HANDSHAKE
    assert len(wire) == 16 + 2 * OUTPUT_FLOATS * 4
    actual = np.frombuffer(wire[16:], dtype="<f4").reshape(2, OUTPUT_FLOATS)
    with torch.inference_mode():
        for index in range(2):
            output = model(features[index : index + 1])
            expected = torch.cat((output.policy_logits, output.value_logits), dim=1)
            np.testing.assert_array_equal(actual[index], expected[0].numpy())
    for name, parameter in model.named_parameters():
        assert parameter.grad is None
        assert torch.equal(parameter, weights_before[name])


@pytest.mark.parametrize("payload", [b"\0", bytes(INPUT_FLOATS * 4 - 1)])
def test_truncated_request_fails_without_a_partial_reply(payload):
    destination = io.BytesIO()
    with pytest.raises(EOFError, match="truncated"):
        serve(small_model(), io.BytesIO(payload), destination)
    assert destination.getvalue() == HANDSHAKE


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_fails_without_a_reply(value):
    features = np.zeros(INPUT_FLOATS, dtype="<f4")
    features[117] = value
    destination = io.BytesIO()
    with pytest.raises(ValueError, match="features must be finite"):
        serve(small_model(), io.BytesIO(features.tobytes()), destination)
    assert destination.getvalue() == HANDSHAKE


@pytest.mark.parametrize("wrong_shape", [False, True])
def test_invalid_model_output_fails_without_a_partial_reply(wrong_shape):
    model = small_model()

    def invalid_output(module, inputs, output):
        if wrong_shape:
            return ModelOutput(output.policy_logits[:, :-1], output.value_logits)
        return ModelOutput(output.policy_logits * float("nan"), output.value_logits)

    hook = model.register_forward_hook(invalid_output)
    destination = io.BytesIO()
    try:
        with pytest.raises(ValueError, match="incompatible|nonfinite"):
            serve(model, io.BytesIO(bytes(INPUT_FLOATS * 4)), destination)
    finally:
        hook.remove()
    assert destination.getvalue() == HANDSHAKE


def test_clean_eof_between_requests_exits():
    destination = io.BytesIO()
    serve(small_model(), io.BytesIO(), destination)
    assert destination.getvalue() == HANDSHAKE


@pytest.mark.parametrize(
    "message", ["checkpoint missing", "é" * 3000, "x" * 4095 + "♞", ""]
)
def test_startup_error_is_bounded_valid_utf8_and_never_a_success_handshake(message):
    destination = io.BytesIO()
    write_startup_error(destination, message)
    data = destination.getvalue()
    magic, version, length, reserved = struct.unpack("<4sIII", data[:16])
    assert (magic, version, reserved) == (b"ODNE", 1, 0)
    assert 0 < length <= 4096
    assert len(data) == 16 + length
    text = data[16:].decode("utf-8")
    assert message.startswith(text) if message else text == "unknown startup error"
