import numpy as np

from gate2_experiment import (
    assert_gate,
    summarize,
)
from serialized_residue import (
    FrozenSequenceReader,
)


def test_same_multiset_different_order_changes_operator():
    reader = FrozenSequenceReader(seed=4)
    rng = np.random.default_rng(44)

    a, b, x, y, c = reader.token_set(rng)

    operator_ab, _ = reader.effective_operator(
        [a, x, b, y, c]
    )
    operator_ba, _ = reader.effective_operator(
        [b, x, a, y, c]
    )

    assert not np.allclose(
        operator_ab,
        operator_ba,
    )


def test_orderless_bag_is_identical_for_matched_sequences():
    reader = FrozenSequenceReader(seed=5)
    rng = np.random.default_rng(55)

    a, b, x, y, c = reader.token_set(rng)

    positive = reader.bag_response(
        [a, x, b, y, c]
    )
    negative = reader.bag_response(
        [b, x, a, y, c]
    )

    assert np.allclose(
        positive,
        negative,
    )


def test_replay_is_deterministic_given_same_reader_state():
    reader = FrozenSequenceReader(seed=6)
    rng = np.random.default_rng(66)

    a, b, x, y, c = reader.token_set(rng)
    sequence = [a, x, b, y, c]

    operator_a, _ = reader.effective_operator(
        sequence
    )
    operator_b, _ = reader.effective_operator(
        sequence
    )

    assert np.allclose(
        operator_a,
        operator_b,
    )


def test_same_sequence_different_prior_state_changes_final_operator():
    reader = FrozenSequenceReader(seed=7)
    rng = np.random.default_rng(77)

    a, b, x, y, c = reader.token_set(rng)
    sequence = [a, x, b, y, c]

    operator_a, _ = reader.effective_operator(
        sequence,
        initial_state=1.5 * reader.context_axis,
    )
    operator_b, _ = reader.effective_operator(
        sequence,
        initial_state=-1.5 * reader.context_axis,
    )

    assert not np.allclose(
        operator_a,
        operator_b,
    )


def test_gate2_receipt():
    receipt = summarize(seeds=16)
    assert_gate(receipt)
