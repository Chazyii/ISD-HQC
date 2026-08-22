import pytest

from isd_hqc.algorithms.bjmm import is_valid_representation


def test_is_valid_representation():
    target_vector = [
        1, 1, 0, 0,
    ]

    left_vector = [
        1, 0, 1, 0,
    ]

    right_vector = [
        0, 1, 1, 0,
    ]

    result = is_valid_representation(
        target_vector=target_vector,
        left_vector=left_vector,
        right_vector=right_vector,
        component_weight=2,
    )

    assert result is True


def test_is_valid_representation_uses_cancellation():
    target_vector = [
        1, 1, 0, 0,
    ]

    left_vector = [
        1, 0, 1, 0,
    ]

    right_vector = [
        0, 1, 1, 0,
    ]

    assert left_vector[2] == 1
    assert right_vector[2] == 1

    assert is_valid_representation(
        target_vector=target_vector,
        left_vector=left_vector,
        right_vector=right_vector,
        component_weight=2,
    )


def test_is_valid_representation_rejects_wrong_xor():
    result = is_valid_representation(
        target_vector=[1, 1, 0, 0],
        left_vector=[1, 0, 1, 0],
        right_vector=[0, 0, 1, 1],
        component_weight=2,
    )

    assert result is False


def test_is_valid_representation_rejects_wrong_component_weight():
    result = is_valid_representation(
        target_vector=[1, 1, 0, 0],
        left_vector=[1, 0, 0, 0],
        right_vector=[0, 1, 0, 0],
        component_weight=2,
    )

    assert result is False


def test_is_valid_representation_rejects_left_length_mismatch():
    with pytest.raises(
        ValueError,
        match="Left vector length must match target vector length.",
    ):
        is_valid_representation(
            target_vector=[1, 1, 0],
            left_vector=[1, 0],
            right_vector=[0, 1, 0],
            component_weight=1,
        )


def test_is_valid_representation_rejects_right_length_mismatch():
    with pytest.raises(
        ValueError,
        match="Right vector length must match target vector length.",
    ):
        is_valid_representation(
            target_vector=[1, 1, 0],
            left_vector=[1, 0, 0],
            right_vector=[0, 1],
            component_weight=1,
        )


def test_is_valid_representation_rejects_negative_component_weight():
    with pytest.raises(
        ValueError,
        match="Component weight must not be negative.",
    ):
        is_valid_representation(
            target_vector=[1, 1],
            left_vector=[1, 0],
            right_vector=[0, 1],
            component_weight=-1,
        )