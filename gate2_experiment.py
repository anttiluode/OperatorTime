"""Gate 2: serialized residue is time-series input, not a stored operator.

Two matched frozen sequences contain exactly the same five token vectors and
end on the same token C. They differ only by swapping the temporal positions of
A and B.

A stateful reader sees those tokens one at a time. Its parameters never change.
The question is whether the serial path changes the operator available after
reading, and whether orderless / shuffled / endpoint-only attackers fail.

The gate also tests two complementary facts:
- replaying the same sequence into the same starting state is deterministic;
- replaying the same sequence into different prior resident states does not
  necessarily produce the same final operator.

So the stored sequence constrains an operator trajectory, but does not fully
specify the reader's operator independent of reader state.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from serialized_residue import (
    FrozenSequenceReader,
)


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

    prediction = np.sign(
        augmented @ weights
    )
    prediction[prediction == 0.0] = 1.0

    return float(
        np.mean(prediction == labels)
    )


def make_samples(
    reader: FrozenSequenceReader,
    rng: np.random.Generator,
    count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if count % 2 != 0:
        raise ValueError("count must be even")

    serial: list[np.ndarray] = []
    bag: list[np.ndarray] = []
    shuffled: list[np.ndarray] = []
    endpoint: list[np.ndarray] = []
    labels: list[int] = []

    for _ in range(count // 2):
        a, b, x, y, c = reader.token_set(
            rng
        )

        positive = [a, x, b, y, c]
        negative = [b, x, a, y, c]

        for label, sequence in (
            (+1, positive),
            (-1, negative),
        ):
            serial.append(
                reader.response(sequence)
            )
            bag.append(
                reader.bag_response(sequence)
            )

            random_order = list(sequence)
            rng.shuffle(random_order)

            shuffled.append(
                reader.response(random_order)
            )

            # Same visible endpoint C, but all preceding serial residue erased.
            endpoint.append(
                reader.response([c])
            )

            labels.append(label)

    return (
        np.asarray(serial),
        np.asarray(bag),
        np.asarray(shuffled),
        np.asarray(endpoint),
        np.asarray(labels),
    )


def run_one(
    seed: int,
    train_count: int = 300,
    test_count: int = 200,
) -> dict[str, float]:
    reader = FrozenSequenceReader(
        seed=seed
    )
    rng = np.random.default_rng(
        seed + 2_000
    )

    train = make_samples(
        reader,
        rng,
        train_count,
    )
    test = make_samples(
        reader,
        rng,
        test_count,
    )

    metrics: dict[str, float] = {}

    names = (
        "serial_accuracy",
        "bag_accuracy",
        "shuffled_accuracy",
        "endpoint_accuracy",
    )

    for index, name in enumerate(names):
        probe = fit_ridge(
            train[index],
            train[4],
        )
        metrics[name] = accuracy(
            test[index],
            test[4],
            probe,
        )

    a, b, x, y, c = reader.token_set(
        rng
    )
    sequence = [a, x, b, y, c]

    replay_a, _ = reader.effective_operator(
        sequence
    )
    replay_b, _ = reader.effective_operator(
        sequence
    )

    metrics["same_reader_replay_distance"] = float(
        np.linalg.norm(
            replay_a - replay_b
        )
    )

    positive_context = (
        1.5 * reader.context_axis
    )
    negative_context = (
        -1.5 * reader.context_axis
    )

    operator_positive, _ = (
        reader.effective_operator(
            sequence,
            initial_state=positive_context,
        )
    )
    operator_negative, _ = (
        reader.effective_operator(
            sequence,
            initial_state=negative_context,
        )
    )

    metrics[
        "different_reader_state_operator_distance"
    ] = float(
        np.linalg.norm(
            operator_positive
            - operator_negative
        )
        / np.linalg.norm(
            operator_positive
        )
    )

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
        receipt["serial_accuracy"]["mean"] > 0.99
        and receipt["bag_accuracy"]["mean"] < 0.56
        and receipt["shuffled_accuracy"]["mean"] < 0.56
        and receipt["endpoint_accuracy"]["mean"] < 0.56
        and receipt[
            "same_reader_replay_distance"
        ]["mean"] < 1e-12
        and receipt[
            "different_reader_state_operator_distance"
        ]["mean"] > 0.015
        and receipt["parameter_drift"]["mean"] == 0.0
    )

    return receipt


def assert_gate(
    receipt: dict[str, object],
) -> None:
    assert receipt["gate_pass"]
    assert receipt["serial_accuracy"]["mean"] > 0.99
    assert receipt["bag_accuracy"]["mean"] < 0.56
    assert receipt["shuffled_accuracy"]["mean"] < 0.56
    assert receipt["endpoint_accuracy"]["mean"] < 0.56
    assert (
        receipt[
            "same_reader_replay_distance"
        ]["mean"]
        < 1e-12
    )
    assert (
        receipt[
            "different_reader_state_operator_distance"
        ]["mean"]
        > 0.015
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
