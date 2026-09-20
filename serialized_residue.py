"""Gate 2: frozen serial input induces an operator trajectory.

A stored sequence is not itself treated as an operator. It is replayed as a
time series into one fixed reader. The reader's resident state changes while
the sequence is consumed, and therefore the effective operator available after
the sequence depends on order.

The same token multiset is used in both classes:
    A -> X -> B -> Y -> C
    B -> X -> A -> Y -> C

The final token C is identical. Only temporal order differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector.copy() if norm == 0.0 else vector / norm


@dataclass
class FrozenSequenceReader:
    seed: int = 0
    dim: int = 32
    basis_count: int = 8
    decay: float = 0.72

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.self_anchor = unit(rng.normal(size=self.dim))

        semantic = unit(rng.normal(size=self.dim))
        semantic -= self.self_anchor * float(
            self.self_anchor @ semantic
        )
        self.semantic_axis = unit(semantic)

        context = unit(rng.normal(size=self.dim))
        context -= self.self_anchor * float(
            self.self_anchor @ context
        )
        context -= self.semantic_axis * float(
            self.semantic_axis @ context
        )
        self.context_axis = unit(context)

        self.base_operator, _ = np.linalg.qr(
            rng.normal(size=(self.dim, self.dim))
        )

        self.current_probe = unit(
            rng.normal(size=self.dim)
        )

        visible_output = unit(
            rng.normal(size=self.dim)
        )

        basis = [
            np.outer(
                visible_output,
                self.current_probe,
            )
        ]
        routing = [self.semantic_axis]

        for _ in range(self.basis_count - 1):
            basis.append(
                np.outer(
                    unit(rng.normal(size=self.dim)),
                    unit(rng.normal(size=self.dim)),
                )
            )
            routing.append(
                unit(rng.normal(size=self.dim))
            )

        self.operator_basis = np.asarray(basis)
        self.routing = np.asarray(routing)

    def token_set(
        self,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, ...]:
        """Create one matched five-token set.

        A and B oppose each other on one latent axis. X/Y/C are nuisance
        tokens projected away from that axis.
        """
        a = unit(
            1.5 * self.semantic_axis
            + 0.45
            * rng.normal(size=self.dim)
            / np.sqrt(self.dim)
        )
        b = unit(
            -1.5 * self.semantic_axis
            + 0.45
            * rng.normal(size=self.dim)
            / np.sqrt(self.dim)
        )

        neutral = []

        for _ in range(3):
            token = rng.normal(size=self.dim)
            token -= self.semantic_axis * float(
                self.semantic_axis @ token
            )
            neutral.append(unit(token))

        x, y, c = neutral
        return a, b, x, y, c

    def resident_state(
        self,
        sequence: list[np.ndarray],
        initial_state: np.ndarray | None = None,
    ) -> np.ndarray:
        state = (
            np.zeros(self.dim, dtype=float)
            if initial_state is None
            else np.asarray(
                initial_state,
                dtype=float,
            ).copy()
        )

        for token in sequence:
            state = self.decay * state + token

        return state

    def operator_from_state(
        self,
        resident: np.ndarray,
    ) -> np.ndarray:
        routing_state = (
            1.1 * self.self_anchor + resident
        )

        coefficients = np.tanh(
            1.4
            * (self.routing @ routing_state)
        )

        return (
            self.base_operator
            + 0.45
            * np.tensordot(
                coefficients,
                self.operator_basis,
                axes=(0, 0),
            )
        )

    def effective_operator(
        self,
        sequence: list[np.ndarray],
        initial_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        resident = self.resident_state(
            sequence,
            initial_state=initial_state,
        )

        return (
            self.operator_from_state(resident),
            resident,
        )

    def response(
        self,
        sequence: list[np.ndarray],
        initial_state: np.ndarray | None = None,
    ) -> np.ndarray:
        operator, _ = self.effective_operator(
            sequence,
            initial_state=initial_state,
        )

        return operator @ self.current_probe

    def bag_response(
        self,
        sequence: list[np.ndarray],
    ) -> np.ndarray:
        """Orderless attacker using exactly the same token multiset."""
        resident = np.sum(
            np.asarray(sequence),
            axis=0,
        )
        operator = self.operator_from_state(
            resident
        )

        return operator @ self.current_probe
