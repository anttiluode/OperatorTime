"""Gate 3: two branch residues can synthesize a new effective operator.

The gate is deliberately narrow. Two counterfactual branches leave residues in
separate provenance subspaces. Neither branch alone contains enough information
for the task. A fixed nonlinear operator factory includes a bilinear interaction
between the two stamped residues. When both are resident together, that
interaction activates an operator component that is absent from either branch
alone and absent from a linear average of their operators.

This demonstrates a mathematical possibility, not spontaneous invention. The
interaction rule is designed by us. A later gate must discover useful
interactions rather than receiving them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector.copy() if norm == 0.0 else vector / norm


@dataclass
class ResidueSynthesisWorld:
    seed: int = 0
    dim: int = 24
    noise: float = 0.08

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        q, _ = np.linalg.qr(
            rng.normal(size=(self.dim, self.dim))
        )

        # Branch-specific physical subspaces act as provenance stamps. The
        # operator factory receives only the resident vector, not a string ID.
        self.branch_a_axis = q[:, 0]
        self.branch_b_axis = q[:, 1]
        self.unstamped_axis = q[:, 2]

        self.current_probe = q[:, 3]
        self.self_anchor = q[:, 7]

        self.base_operator, _ = np.linalg.qr(
            rng.normal(size=(self.dim, self.dim))
        )

        # Make three operator components independently visible on the same
        # current probe: branch A, branch B, and their nonlinear conjunction.
        self.branch_a_operator = np.outer(
            q[:, 4],
            self.current_probe,
        )
        self.branch_b_operator = np.outer(
            q[:, 5],
            self.current_probe,
        )
        self.synergy_operator = np.outer(
            q[:, 6],
            self.current_probe,
        )

    def branch_residue(
        self,
        sign: int,
        branch: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 or +1")

        if branch == "A":
            axis = self.branch_a_axis
        elif branch == "B":
            axis = self.branch_b_axis
        else:
            raise ValueError("branch must be A or B")

        noise = rng.normal(size=self.dim)

        # Keep nuisance noise out of the stamped axes so the matched task is
        # about composition rather than accidental cross-talk.
        for protected in (
            self.branch_a_axis,
            self.branch_b_axis,
            self.unstamped_axis,
        ):
            noise -= protected * float(
                protected @ noise
            )

        return (
            sign * axis
            + self.noise * unit(noise)
        )

    def effective_operator(
        self,
        resident: np.ndarray,
    ) -> np.ndarray:
        a = float(
            self.branch_a_axis @ resident
        )
        b = float(
            self.branch_b_axis @ resident
        )

        branch_a_gain = np.tanh(1.5 * a)
        branch_b_gain = np.tanh(1.5 * b)

        # The key nonlinearity: an interaction term that exists only when both
        # provenance-stamped residues are simultaneously resident.
        synergy_gain = np.tanh(
            2.5 * a * b
        )

        return (
            self.base_operator
            + 0.25
            * branch_a_gain
            * self.branch_a_operator
            + 0.25
            * branch_b_gain
            * self.branch_b_operator
            + 0.50
            * synergy_gain
            * self.synergy_operator
        )

    def response(
        self,
        resident: np.ndarray,
    ) -> np.ndarray:
        return (
            self.effective_operator(resident)
            @ self.current_probe
        )

    def erase_provenance(
        self,
        sign_a: int,
        sign_b: int,
    ) -> np.ndarray:
        """Collapse both branches into one common channel.

        The total signed content survives, but which contribution came from
        which branch does not.
        """
        return (
            sign_a + sign_b
        ) * self.unstamped_axis
