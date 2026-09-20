import numpy as np

from experiment import assert_gate, summarize
from operator_time import OperatorMatter


def test_same_present_can_have_different_effective_operator():
    matter = OperatorMatter(seed=3)
    rng = np.random.default_rng(99)

    object_a = matter.transient_object(+1, rng)
    object_b = matter.transient_object(-1, rng)

    operator_ab, _ = matter.effective_operator(
        [object_a, object_b]
    )
    operator_ba, _ = matter.effective_operator(
        [object_b, object_a]
    )

    # Same substrate, same self anchor, same current probe.
    # Only the recent route differs.
    assert not np.allclose(operator_ab, operator_ba)


def test_erasing_residue_collapses_history_dependence():
    matter = OperatorMatter(seed=4)
    rng = np.random.default_rng(100)

    object_a = matter.transient_object(+1, rng)
    object_b = matter.transient_object(-1, rng)

    operator_ab, _ = matter.effective_operator(
        [object_a, object_b],
        erase_residue=True,
    )
    operator_ba, _ = matter.effective_operator(
        [object_b, object_a],
        erase_residue=True,
    )

    assert np.allclose(operator_ab, operator_ba)


def test_self_anchor_is_not_rewritten_by_transient_objects():
    matter = OperatorMatter(seed=5)
    original = matter.self_anchor.copy()
    rng = np.random.default_rng(101)

    history = [
        matter.transient_object(
            int(rng.choice(np.array([-1, 1]))),
            rng,
        )
        for _ in range(20)
    ]

    matter.effective_operator(history)

    assert np.allclose(
        matter.self_anchor,
        original,
    )


def test_gate1_receipt():
    receipt = summarize(seeds=16)
    assert_gate(receipt)
