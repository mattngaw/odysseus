import pytest
import torch

from neurodiktyon import BASE_MOVE_COUNT, POLICY_SIZE, PolicyMap


def square_name(index):
    rank, file = divmod(index, 8)
    return f"{chr(ord('a') + file)}{rank + 1}"


def labels_from_gathered_identifiers(output):
    labels = []
    for index, value in enumerate(output.tolist()):
        if index < 1792:
            source, destination = divmod(int(value), 100)
            labels.append(square_name(source) + square_name(destination))
        else:
            source_file, rest = divmod(int(value) - 10_000, 100)
            destination_file, piece = divmod(rest, 10)
            labels.append(
                f"{chr(ord('a') + source_file)}7"
                f"{chr(ord('a') + destination_file)}8{'qrb'[piece]}"
            )
    return labels


def identified_inputs():
    source = torch.arange(64, dtype=torch.float64)
    pairs = source[:, None] * 100 + source[None, :]
    files = torch.arange(8, dtype=torch.float64)
    promotions = (
        10_000
        + files[:, None, None] * 100
        + files[None, :, None] * 10
        + torch.arange(3, dtype=torch.float64)
    )
    return torch.stack([pairs, -pairs]), torch.stack([promotions, -promotions])


def test_every_gathered_entry_matches_the_frozen_pyxis_vocabulary_contract():
    mapping = PolicyMap()
    pairs, promotions = identified_inputs()
    output = mapping(pairs, promotions)
    assert (BASE_MOVE_COUNT, POLICY_SIZE) == (1792, 1858)
    assert output.shape == (2, 1858)
    assert torch.equal(output[1], -output[0])
    labels = labels_from_gathered_identifiers(output[0])
    assert len(set(labels)) == 1858

    # Independent ordering fingerprint frozen in crates/pyxis/tests/vocabulary.rs.
    checksum = 0xCBF29CE484222325
    for byte in ("\n".join(labels) + "\n").encode():
        checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    assert checksum == 0x32DCAF215253E599
    for index, label in [
        (0, "a1b1"),
        (97, "e1a1"),
        (102, "e1g1"),
        (103, "e1h1"),
        (159, "g1f3"),
        (322, "e2e4"),
        (1401, "a7a8"),
        (1791, "h8g8"),
        (1792, "a7a8q"),
        (1793, "a7a8r"),
        (1794, "a7a8b"),
        (1857, "h7h8b"),
    ]:
        assert labels[index] == label


def test_gradients_route_to_selected_cells_and_leave_all_other_cells_zero():
    pairs, promotions = identified_inputs()
    # Decode labels from sentinel values, independently of the map's index buffers.
    mapping = PolicyMap()
    labels = labels_from_gathered_identifiers(mapping(pairs, promotions)[0])
    pairs.requires_grad_()
    promotions.requires_grad_()
    output = mapping(pairs, promotions)
    upstream = torch.arange(1, 1859, dtype=torch.float64).repeat(2, 1)
    upstream[1].neg_()
    (output * upstream).sum().backward()

    expected_pairs = torch.zeros_like(pairs)
    expected_promotions = torch.zeros_like(promotions)
    for index, label in enumerate(labels):
        source_file = ord(label[0]) - ord("a")
        destination_file = ord(label[2]) - ord("a")
        if len(label) == 4:
            source = (int(label[1]) - 1) * 8 + source_file
            destination = (int(label[3]) - 1) * 8 + destination_file
            expected_pairs[:, source, destination] = upstream[:, index]
        else:
            expected_promotions[
                :, source_file, destination_file, "qrb".index(label[4])
            ] = upstream[:, index]
    assert torch.equal(pairs.grad, expected_pairs)
    assert torch.equal(promotions.grad, expected_promotions)


def test_noncontiguous_inputs_and_empty_batches():
    mapping = PolicyMap()
    pairs = torch.randn(64, 2, 64).transpose(0, 1)
    promotions = torch.randn(8, 2, 8, 3).transpose(0, 1)
    assert not pairs.is_contiguous() and not promotions.is_contiguous()
    torch.testing.assert_close(
        mapping(pairs, promotions), mapping(pairs.contiguous(), promotions.contiguous())
    )
    assert mapping(pairs[:0], promotions[:0]).shape == (0, 1858)


def test_indices_are_registered_integer_buffers_and_survive_state_round_trip():
    mapping = PolicyMap().to(dtype=torch.float64)
    assert list(mapping.parameters()) == []
    assert set(dict(mapping.named_buffers())) == {"base_indices", "promotion_indices"}
    assert all(buffer.dtype == torch.long for buffer in mapping.buffers())
    restored = PolicyMap()
    restored.load_state_dict(mapping.state_dict())
    pairs, promotions = identified_inputs()
    assert torch.equal(restored(pairs, promotions), mapping(pairs, promotions))
    # Moving buffers can be checked without requiring a GPU on the test machine.
    mapping.to("meta")
    assert all(buffer.device.type == "meta" for buffer in mapping.buffers())


@pytest.mark.parametrize("shape", [(64, 64), (2, 63, 64), (2, 64, 63)])
def test_rejects_incompatible_pair_layout(shape):
    with pytest.raises(ValueError, match=r"expected pair_logits \[batch, 64, 64\]"):
        PolicyMap()(torch.zeros(shape), torch.zeros(2, 8, 8, 3))


@pytest.mark.parametrize("shape", [(8, 8, 3), (1, 8, 8, 3), (2, 8, 8, 4), (2, 7, 8, 3)])
def test_rejects_promotion_layout_or_batch_mismatch(shape):
    with pytest.raises(ValueError, match="expected promotion_logits shape"):
        PolicyMap()(torch.zeros(2, 64, 64), torch.zeros(shape))
