"""Minimal fixed-substrate model for the Operator Time idea.

The learned/substrate parameters never change. Transient objects update a
resident state, and that resident state changes which mixture of fixed basis
operators acts on an identical current probe.

This is a mechanism toy, not a transformer implementation or biological model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector.copy() if norm == 0.0 else vector / norm


@dataclass
class OperatorMatter:
    seed: int = 0
    dim: int = 32
    basis_count: int = 10
    decay: float = 0.72

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.self_anchor = unit(rng.normal(size=self.dim))

        semantic = unit(rng.normal(size=self.dim))
        semantic -= self.self_anchor * float(
            self.self_anchor @ semantic
        )
        self.semantic_axis = unit(semantic)

        # Fixed base transport.
        self.base_operator, _ = np.linalg.qr(
            rng.normal(size=(self.dim, self.dim))
        )

        # Fixed low-rank operator basis.
        basis = []
        for _ in range(self.basis_count):
            left = unit(rng.normal(size=self.dim))
            right = unit(rng.normal(size=self.dim))
            basis.append(np.outer(left, right))
        self.operator_basis = np.asarray(basis)

        # Fixed state-to-operator routing directions. The first direction is
        # aligned with a latent object feature so the accessibility of that
        # feature can be measured by an external probe.
        self.routing = np.stack(
            [self.semantic_axis]
            + [
                unit(rng.normal(size=self.dim))
                for _ in range(self.basis_count - 1)
            ]
        )

        # The current probe is exactly the same regardless of history.
        self.current_probe = unit(rng.normal(size=self.dim))

        # Ensure at least one basis operator is clearly visible to the probe.
        visible_output = unit(rng.normal(size=self.dim))
        self.operator_basis[0] = np.outer(
            visible_output,
            self.current_probe,
        )

    def transient_object(
        self,
        label: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate a new object instance from one of two latent classes."""
        if label not in (-1, 1):
            raise ValueError("label must be -1 or +1")

        noise = (
            0.7
            * rng.normal(size=self.dim)
            / np.sqrt(self.dim)
        )

        return unit(
            1.3 * label * self.semantic_axis + noise
        )

    def resident_state(
        self,
        history: list[np.ndarray],
    ) -> np.ndarray:
        """Write transient objects into a decaying resident state."""
        state = np.zeros(self.dim, dtype=float)

        for object_vector in history:
            state = self.decay * state + object_vector

        # One neutral return-to-SELF tick: the transient object is gone, but
        # its residue has not vanished.
        return self.decay * state

    def effective_operator(
        self,
        history: list[np.ndarray],
        erase_residue: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the effective operator from fixed substrate + state."""
        resident = self.resident_state(history)

        if erase_residue:
            resident = np.zeros_like(resident)

        # SELF is always present as an anchor. The recent path changes the
        # operator mixture without changing any substrate parameter.
        routing_state = 1.1 * self.self_anchor + resident

        coefficients = np.tanh(
            1.4 * (self.routing @ routing_state)
        )

        operator = self.base_operator + 0.45 * np.tensordot(
            coefficients,
            self.operator_basis,
            axes=(0, 0),
        )

        return operator, resident

    def response(
        self,
        history: list[np.ndarray],
        erase_residue: bool = False,
    ) -> np.ndarray:
        operator, _ = self.effective_operator(
            history,
            erase_residue=erase_residue,
        )
        return operator @ self.current_probe

    def static_response(self) -> np.ndarray:
        """Attacker: one fixed operator, no resident-state dependence."""
        return self.base_operator @ self.current_probe
