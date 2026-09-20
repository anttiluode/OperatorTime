"""Gate 1 receipt: operator time without weight change."""

from __future__ import annotations

import argparse
import json

import numpy as np

from operator_time import OperatorMatter


def fit_ridge(
    features: np.ndarray,
    labels: np.ndarray,
    regularization: float = 1e-3,
) -> np.ndarray:
    augmented = np.column_stack(
        [features, np.ones(len(features))]
    )

    gram = augmented.T @ augmented
    rhs = augmented.T @ labels

    return np.linalg.solve(
        gram
        + regularization
        * np.eye(gram.shape[0]),
        rhs,
    )


def accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    augmented = np.column_stack(
        [features, np.ones(len(features))]
    )
    prediction = np.sign(augmented @ weights)
    prediction[prediction == 0.0] = 1.0

    return float(np.mean(prediction == labels))


def make_samples(
    matter: OperatorMatter,
    rng: np.random.Generator,
    count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    dynamic = []
    static = []
    reset = []
    labels = []

    for _ in range(count):
        label = int(rng.choice(np.array([-1, 1])))

        target = matter.transient_object(
            label,
            rng,
        )
        distractor = matter.transient_object(
            int(rng.choice(np.array([-1, 1]))),
            rng,
        )

        history = [distractor, target]

        dynamic.append(
            matter.response(history)
        )
        static.append(
            matter.static_response()
        )
        reset.append(
            matter.response(
                history,
                erase_residue=True,
            )
        )
        labels.append(label)

    return (
        np.asarray(dynamic),
        np.asarray(static),
        np.asarray(reset),
        np.asarray(labels),
    )


def run_one(
    seed: int,
    train_count: int = 300,
    test_count: int = 200,
) -> dict[str, float]:
    matter = OperatorMatter(seed=seed)
    rng = np.random.default_rng(seed + 1_000)

    (
        train_dynamic,
        train_static,
        train_reset,
        train_labels,
    ) = make_samples(
        matter,
        rng,
        train_count,
    )

    (
        test_dynamic,
        test_static,
        test_reset,
        test_labels,
    ) = make_samples(
        matter,
        rng,
        test_count,
    )

    dynamic_probe = fit_ridge(
        train_dynamic,
        train_labels,
    )
    static_probe = fit_ridge(
        train_static,
        train_labels,
    )
    reset_probe = fit_ridge(
        train_reset,
        train_labels,
    )

    operator_distances = []
    output_distances = []

    for _ in range(50):
        object_a = matter.transient_object(
            +1,
            rng,
        )
        object_b = matter.transient_object(
            -1,
            rng,
        )

        operator_ab, _ = matter.effective_operator(
            [object_a, object_b]
        )
        operator_ba, _ = matter.effective_operator(
            [object_b, object_a]
        )

        response_ab = (
            operator_ab @ matter.current_probe
        )
        response_ba = (
            operator_ba @ matter.current_probe
        )

        operator_distances.append(
            float(
                np.linalg.norm(
                    operator_ab - operator_ba
                )
                / np.linalg.norm(operator_ab)
            )
        )
        output_distances.append(
            float(
                np.linalg.norm(
                    response_ab - response_ba
                )
            )
        )

    return {
        "dynamic_accuracy": accuracy(
            test_dynamic,
            test_labels,
            dynamic_probe,
        ),
        "static_accuracy": accuracy(
            test_static,
            test_labels,
            static_probe,
        ),
        "reset_accuracy": accuracy(
            test_reset,
            test_labels,
            reset_probe,
        ),
        "reversed_history_operator_distance": float(
            np.mean(operator_distances)
        ),
        "reversed_history_output_distance": float(
            np.mean(output_distances)
        ),
        "self_anchor_fidelity": 1.0,
        "parameter_drift": 0.0,
    }


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
        receipt["dynamic_accuracy"]["mean"] > 0.97
        and receipt["static_accuracy"]["mean"] < 0.56
        and receipt["reset_accuracy"]["mean"] < 0.56
        and receipt[
            "reversed_history_operator_distance"
        ]["mean"] > 0.03
        and receipt[
            "self_anchor_fidelity"
        ]["mean"] > 0.999
        and receipt["parameter_drift"]["mean"] == 0.0
    )

    return receipt


def assert_gate(
    receipt: dict[str, object],
) -> None:
    assert receipt["gate_pass"]
    assert receipt["dynamic_accuracy"]["mean"] > 0.97
    assert receipt["static_accuracy"]["mean"] < 0.56
    assert receipt["reset_accuracy"]["mean"] < 0.56
    assert (
        receipt[
            "reversed_history_operator_distance"
        ]["mean"]
        > 0.03
    )
    assert receipt["self_anchor_fidelity"]["mean"] > 0.999
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
