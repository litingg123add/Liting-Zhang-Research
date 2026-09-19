"""Dynamical-system diagnostics for the reduced LNETLM model."""

from __future__ import annotations

import numpy as np

from lnetlm import LNETLMParameters, rhs
from rk38 import integrate, rk38_step


def finite_time_lyapunov(
    y0: np.ndarray,
    p: LNETLMParameters,
    *,
    t_span: tuple[float, float] = (0.0, 50.0),
    step: float = 0.01,
    perturbation: float = 1e-8,
    renormalise_every: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a largest finite-time trajectory-separation exponent.

    The two nearby trajectories receive the same periodic forcing. This function
    is intended as a numerical diagnostic, not as a stand-alone proof of chaos.
    """
    y = np.asarray(y0, dtype=float).copy()
    direction = np.array([1.0, 0.0])
    y_perturbed = y + perturbation * direction

    total_steps = int(np.floor((t_span[1] - t_span[0]) / step))
    sample_t = []
    estimates = []
    accumulated = 0.0
    t = t_span[0]

    fun = lambda time, state: rhs(time, state, p)

    for i in range(1, total_steps + 1):
        y = rk38_step(fun, t, y, step)
        y_perturbed = rk38_step(fun, t, y_perturbed, step)
        t += step

        if i % renormalise_every == 0:
            delta = y_perturbed - y
            distance = float(np.linalg.norm(delta))
            if not np.isfinite(distance) or distance == 0.0:
                break
            accumulated += np.log(distance / perturbation)
            elapsed = t - t_span[0]
            sample_t.append(t)
            estimates.append(accumulated / elapsed)

            delta = delta / distance * perturbation
            y_perturbed = y + delta

        if np.linalg.norm(y) > 1e6 or np.linalg.norm(y_perturbed) > 1e6:
            break

    return np.asarray(sample_t), np.asarray(estimates)


def stroboscopic_section(
    y0: np.ndarray,
    p: LNETLMParameters,
    *,
    transient_periods: int = 50,
    keep_periods: int = 150,
    steps_per_period: int = 300,
) -> np.ndarray:
    """Sample a periodically forced trajectory once per forcing period."""
    if p.A == 0 or p.omega == 0:
        raise ValueError("Stroboscopic sections require non-zero periodic forcing.")

    period = 2.0 * np.pi / abs(p.omega)
    step = period / steps_per_period
    total_periods = transient_periods + keep_periods

    times, states = integrate(
        lambda t, y: rhs(t, y, p),
        y0,
        (0.0, total_periods * period),
        step,
        divergence_limit=1e6,
    )

    if len(states) < total_periods * steps_per_period + 1:
        return np.empty((0, 2))

    indices = np.arange(
        transient_periods * steps_per_period,
        total_periods * steps_per_period + 1,
        steps_per_period,
    )
    return states[indices]


def trajectory_separation(
    y0: np.ndarray,
    y1: np.ndarray,
    p: LNETLMParameters,
    *,
    t_span: tuple[float, float] = (0.0, 20.0),
    step: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Euclidean separation between two trajectories."""
    fun = lambda t, y: rhs(t, y, p)
    t0, x0 = integrate(fun, y0, t_span, step, divergence_limit=1e6)
    t1, x1 = integrate(fun, y1, t_span, step, divergence_limit=1e6)
    n = min(len(t0), len(t1))
    return t0[:n], np.linalg.norm(x0[:n] - x1[:n], axis=1)
