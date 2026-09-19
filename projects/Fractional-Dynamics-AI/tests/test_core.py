import numpy as np

from src.lnetlm import LNETLMParameters, equilibrium_report, rhs
from src.rk38 import integrate


def test_equilibrium_classification():
    p = LNETLMParameters(P=1.0, T=1.0, Q=-2.0, c=4.0)
    report = equilibrium_report(p)
    labels = {(round(r["U"], 8), r["classification"]) for r in report}
    assert (0.0, "saddle") in labels
    assert (-2.0, "asymptotically stable") in labels


def test_reference_trajectory_converges_to_stable_equilibrium():
    p = LNETLMParameters(P=1.0, T=1.0, Q=-2.0, c=4.0)
    _, state = integrate(
        lambda t, y: rhs(t, y, p),
        np.array([-3.0, 0.5]),
        (0.0, 20.0),
        0.01,
    )
    assert abs(state[-1, 0] + 2.0) < 2e-3
    assert abs(state[-1, 1]) < 2e-3
