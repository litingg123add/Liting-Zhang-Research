"""Reduced dynamical system for the lossy nonlinear electrical transmission line model.

The travelling-wave equation used in the associated research is

    P U'' + T U' + Q U^2 - c U = 0,

which can be written as the planar first-order system

    U' = V
    V' = -(T/P)V - (Q/P)U^2 + (c/P)U.

A periodic forcing term A*sin(omega*t) can be enabled for sensitivity studies.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LNETLMParameters:
    P: float = 1.0
    T: float = 1.0
    Q: float = -2.0
    c: float = 4.0
    A: float = 0.0
    omega: float = 2.0 * np.pi

    def validate(self) -> None:
        if self.P == 0:
            raise ValueError("P must be non-zero.")


def rhs(t: float, state: np.ndarray, p: LNETLMParameters) -> np.ndarray:
    """Right-hand side of the reduced LNETLM system."""
    p.validate()
    u, v = np.asarray(state, dtype=float)
    forcing = p.A * np.sin(p.omega * t)
    return np.array(
        [
            v,
            -(p.T / p.P) * v
            - (p.Q / p.P) * u**2
            + (p.c / p.P) * u
            + forcing,
        ],
        dtype=float,
    )


def jacobian(state: np.ndarray, p: LNETLMParameters) -> np.ndarray:
    """Jacobian with respect to (U, V)."""
    p.validate()
    u, _ = np.asarray(state, dtype=float)
    return np.array(
        [
            [0.0, 1.0],
            [(p.c - 2.0 * p.Q * u) / p.P, -p.T / p.P],
        ],
        dtype=float,
    )


def equilibria(p: LNETLMParameters) -> list[np.ndarray]:
    """Return equilibria of the unforced planar system."""
    p.validate()
    if p.A != 0:
        raise ValueError("Time-periodic forcing removes fixed equilibria in the 2D system.")
    roots = [np.array([0.0, 0.0])]
    if p.Q != 0:
        second = p.c / p.Q
        if not np.isclose(second, 0.0):
            roots.append(np.array([second, 0.0]))
    return roots


def equilibrium_report(p: LNETLMParameters) -> list[dict]:
    """Return eigenvalue-based local stability information."""
    report = []
    for point in equilibria(p):
        eig = np.linalg.eigvals(jacobian(point, p))
        real = np.real(eig)
        if np.all(real < 0):
            label = "asymptotically stable"
        elif np.any(real > 0) and np.any(real < 0):
            label = "saddle"
        elif np.any(real > 0):
            label = "unstable"
        else:
            label = "non-hyperbolic / marginal"
        report.append(
            {
                "U": float(point[0]),
                "V": float(point[1]),
                "eigenvalues": [complex(value) for value in eig],
                "classification": label,
            }
        )
    return report
