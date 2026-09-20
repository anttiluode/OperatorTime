import numpy as np

from gate3_experiment import (
    assert_gate,
    summarize,
)
from residue_synthesis import (
    ResidueSynthesisWorld,
)


def test_synergy_term_is_absent_from_each_branch_alone():
    world = ResidueSynthesisWorld(seed=9)
    rng = np.random.default_rng(90)

    residue_a = world.branch_residue(
        +1,
        "A",
        rng,
    )
    residue_b = world.branch_residue(
        -1,
        "B",
        rng,
    )

    a_a = float(
        world.branch_a_axis @ residue_a
    )
    b_a = float(
        world.branch_b_axis @ residue_a
    )
    a_b = float(
        world.branch_a_axis @ residue_b
    )
    b_b = float(
        world.branch_b_axis @ residue_b
    )

    assert abs(a_a * b_a) < 1e-12
    assert abs(a_b * b_b) < 1e-12


def test_joint_operator_is_not_linear_average_of_branch_operators():
    world = ResidueSynthesisWorld(seed=10)
    rng = np.random.default_rng(100)

    residue_a = world.branch_residue(
        +1,
        "A",
        rng,
    )
    residue_b = world.branch_residue(
        +1,
        "B",
        rng,
    )

    operator_a = world.effective_operator(
        residue_a
    )
    operator_b = world.effective_operator(
        residue_b
    )
    operator_joint = world.effective_operator(
        residue_a + residue_b
    )

    assert not np.allclose(
        operator_joint,
        0.5 * (operator_a + operator_b),
    )


def test_erasing_provenance_removes_synergy_channel():
    world = ResidueSynthesisWorld(seed=11)

    resident = world.erase_provenance(
        +1,
        -1,
    )

    assert abs(
        float(
            world.branch_a_axis @ resident
        )
    ) < 1e-12
    assert abs(
        float(
            world.branch_b_axis @ resident
        )
    ) < 1e-12


def test_gate3_receipt():
    receipt = summarize(seeds=16)
    assert_gate(receipt)
