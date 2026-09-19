"""Reproduce the public dynamics examples used in the project README."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lnetlm import LNETLMParameters, equilibrium_report, rhs
from rk38 import integrate


def main() -> None:
    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    p = LNETLMParameters(P=1.0, T=1.0, Q=-2.0, c=4.0)
    print("Equilibria:")
    for item in equilibrium_report(p):
        print(item)

    initial_conditions = [
        (-3.0, 0.5),
        (-3.0, -0.5),
        (-1.5, 1.0),
        (-1.2, -1.0),
        (0.4, -1.5),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for initial in initial_conditions:
        _, state = integrate(
            lambda t, y: rhs(t, y, p),
            np.asarray(initial),
            (0.0, 20.0),
            0.01,
            divergence_limit=1e4,
        )
        ax.plot(state[:, 0], state[:, 1], lw=1.1, label=str(initial))

    ax.scatter([0.0, -2.0], [0.0, 0.0], s=45, zorder=5)
    ax.set(xlabel="U", ylabel="V", title="Reduced LNETLM phase portrait")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "phase_portrait.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
