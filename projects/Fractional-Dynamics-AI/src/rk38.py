"""Fourth-order Runge-Kutta 3/8 integrator.

This is the four-stage scheme used in the numerical section of the associated
LNETLM study:

    y_{n+1} = y_n + h/8 * (k1 + 3 k2 + 3 k3 + k4)

with stages at 0, 1/3, 2/3 and 1 of the step.
"""

from __future__ import annotations

from collections.abc import Callable
import numpy as np


Array = np.ndarray
RHS = Callable[[float, Array], Array]


def rk38_step(fun: RHS, t: float, y: Array, h: float) -> Array:
    """Advance one explicit RK 3/8 step."""
    y = np.asarray(y, dtype=float)
    k1 = fun(t, y)
    k2 = fun(t + h / 3.0, y + h * k1 / 3.0)
    k3 = fun(t + 2.0 * h / 3.0, y - h * k1 / 3.0 + h * k2)
    k4 = fun(t + h, y + h * k1 - h * k2 + h * k3)
    return y + (h / 8.0) * (k1 + 3.0 * k2 + 3.0 * k3 + k4)


def integrate(
    fun: RHS,
    y0: Array,
    t_span: tuple[float, float],
    step: float,
    *,
    divergence_limit: float | None = None,
) -> tuple[Array, Array]:
    """Integrate on an equally spaced grid using RK 3/8."""
    t0, t1 = t_span
    if step <= 0 or t1 <= t0:
        raise ValueError("Require step > 0 and t1 > t0.")

    n_steps = int(np.floor((t1 - t0) / step))
    times = t0 + np.arange(n_steps + 1, dtype=float) * step
    states = np.empty((len(times), len(np.asarray(y0))), dtype=float)
    states[0] = np.asarray(y0, dtype=float)

    for i in range(n_steps):
        states[i + 1] = rk38_step(fun, times[i], states[i], step)
        if not np.all(np.isfinite(states[i + 1])):
            return times[: i + 2], states[: i + 2]
        if divergence_limit is not None and np.linalg.norm(states[i + 1]) > divergence_limit:
            return times[: i + 2], states[: i + 2]

    return times, states
