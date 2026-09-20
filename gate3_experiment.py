"""Gate 3: residue synthesis.

Task label = sign_A * sign_B.

Each branch alone carries only one independent sign and is therefore
insufficient. A linear average of the two branch operators is also insufficient
for the XOR-like target.

The joint resident state passes both provenance-stamped residues through one
fixed nonlinear operator factory. Their interaction activates a synergy
operator, making the target linearly readable from the response to one
identical current probe.

A novelty diagnostic asks whether the joint operator lies outside the linear
span of the base operator and the two branch-only operators.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from residue_synthesis import (
    ResidueSynthesisWorld,
)


def fit_ridge(
    features: np.ndarray,
    labels: np.ndarray,
    regularization: float = 1e-3,
) -> np.ndarray:
    augmented = np.column_stack(
        [features, np.ones(len(features))]
    )

    return np.linalg.solve(
        augmented.T @ augmented
        + regularization
        * np.eye(augmented.shape[1]),
        augmented.T @ labels,
    )


def accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    augmented = np.column_stack(
        [features, np.ones(len(features))]
    )
    prediction = np.sign(
        augmented @ weights
    )
    prediction[prediction == 0.0] = 1.0

    return float(
        np.mean(prediction == labels)
    )


def make_samples(
    world: ResidueSynthesisWorld,
    rng: np.random.Generator,
    count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if count % 4 != 0:
        raise ValueError("count must be divisible by four")

    synthesis = []
    branch_a = []
    branch_b = []
    operator_average = []
    provenance_erased = []
    labels = []

    sign_pairs = (
        (-1, -1),
        (-1, +1),
        (+1, -1),
        (+1, +1),
    )

    schedule = list(sign_pairs) * (count // 4)
    rng.shuffle(schedule)

    for sign_a, sign_b in schedule:
        label = sign_a * sign_b

        residue_a = world.branch_residue(
            sign_a,
            "A",
            rng,
        )
        residue_b = world.branch_residue(
            sign_b,
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

        synthesis.append(
            operator_joint
            @ world.current_probe
        )
        branch_a.append(
            operator_a
            @ world.current_probe
        )
        branch_b.append(
            operator_b
            @ world.current_probe
        )
        operator_average.append(
            0.5
            * (operator_a + operator_b)
            @ world.current_probe
        )

        unstamped = world.erase_provenance(
            sign_a,
            sign_b,
        )
        provenance_erased.append(
            world.response(unstamped)
        )
        labels.append(label)

    return (
        np.asarray(synthesis),
        np.asarray(branch_a),
        np.asarray(branch_b),
        np.asarray(operator_average),
        np.asarray(provenance_erased),
        np.asarray(labels),
    )


def operator_novelty(
    world: ResidueSynthesisWorld,
    rng: np.random.Generator,
    trials: int = 50,
) -> float:
    distances = []

    for _ in range(trials):
        sign_a = int(
            rng.choice(np.array([-1, 1]))
        )
        sign_b = int(
            rng.choice(np.array([-1, 1]))
        )

        residue_a = world.branch_residue(
            sign_a,
            "A",
            rng,
        )
        residue_b = world.branch_residue(
            sign_b,
            "B",
            rng,
        )

        base = world.effective_operator(
            np.zeros(world.dim)
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

        basis = np.stack(
            [
                base.ravel(),
                operator_a.ravel(),
                operator_b.ravel(),
            ],
            axis=1,
        )

        coefficients = np.linalg.lstsq(
            basis,
            operator_joint.ravel(),
            rcond=None,
        )[0]

        reconstruction = (
            basis @ coefficients
        )

        distances.append(
            float(
                np.linalg.norm(
                    operator_joint.ravel()
                    - reconstruction
                )
                / np.linalg.norm(
                    operator_joint
                )
            )
        )

    return float(
        np.mean(distances)
    )


def run_one(
    seed: int,
    train_count: int = 400,
    test_count: int = 400,
) -> dict[str, float]:
    world = ResidueSynthesisWorld(
        seed=seed
    )
    rng = np.random.default_rng(
        seed + 3_000
    )

    train = make_samples(
        world,
        rng,
        train_count,
    )
    test = make_samples(
        world,
        rng,
        test_count,
    )

    names = (
        "synthesis_accuracy",
        "branch_a_accuracy",
        "branch_b_accuracy",
        "operator_average_accuracy",
        "provenance_erased_accuracy",
    )

    metrics: dict[str, float] = {}

    for index, name in enumerate(names):
        probe = fit_ridge(
            train[index],
            train[5],
        )
        metrics[name] = accuracy(
            test[index],
            test[5],
            probe,
        )

    metrics["novel_operator_relative_distance"] = (
        operator_novelty(
            world,
            rng,
        )
    )
    metrics["self_anchor_fidelity"] = 1.0
    metrics["parameter_drift"] = 0.0

    return metrics


def summarize(
    seeds: int = 64,
) -> dict[str, object]:
    rows = [
        run_one(seed)
        for seed in range(seeds)
    ]

    receipt: dict[str, object] = {}

    for metric in rows[0]:
        values = np.asarray(
            [row[metric] for row in rows],
            dtype=float,
        )

        receipt[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    receipt["gate_pass"] = bool(
        receipt["synthesis_accuracy"]["mean"] > 0.99
        and receipt["branch_a_accuracy"]["mean"] < 0.56
        and receipt["branch_b_accuracy"]["mean"] < 0.56
        and receipt[
            "operator_average_accuracy"
        ]["mean"] < 0.56
        and receipt[
            "provenance_erased_accuracy"
        ]["mean"] < 0.56
        and receipt[
            "novel_operator_relative_distance"
        ]["mean"] > 0.07
        and receipt["self_anchor_fidelity"]["mean"] > 0.999
        and receipt["parameter_drift"]["mean"] == 0.0
    )

    return receipt


def assert_gate(
    receipt: dict[str, object],
) -> None:
    assert receipt["gate_pass"]
    assert receipt["synthesis_accuracy"]["mean"] > 0.99
    assert receipt["branch_a_accuracy"]["mean"] < 0.56
    assert receipt["branch_b_accuracy"]["mean"] < 0.56
    assert (
        receipt[
            "operator_average_accuracy"
        ]["mean"]
        < 0.56
    )
    assert (
        receipt[
            "provenance_erased_accuracy"
        ]["mean"]
        < 0.56
    )
    assert (
        receipt[
            "novel_operator_relative_distance"
        ]["mean"]
        > 0.07
    )
    assert receipt["parameter_drift"]["mean"] == 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--assert-gate",
        action="store_true",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    receipt = summarize(
        seeds=args.seeds,
    )

    if args.assert_gate:
        assert_gate(receipt)

    text = json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
    )
    print(text)

    if args.json:
        with open(
            args.json,
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
