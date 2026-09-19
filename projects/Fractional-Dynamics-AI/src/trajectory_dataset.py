"""Generate trajectory datasets for data-driven / SciML experiments.

The aim is to support generalisation tests across unseen parameter values,
rather than random point-wise splits that leak information from the same
trajectory into both training and testing.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json

import numpy as np

from lnetlm import LNETLMParameters, rhs
from rk38 import integrate


def make_windows(
    trajectory: np.ndarray,
    input_steps: int,
    forecast_steps: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one trajectory into input/forecast sequence pairs."""
    total = input_steps + forecast_steps
    starts = range(0, len(trajectory) - total + 1, stride)
    x, y = [], []
    for start in starts:
        x.append(trajectory[start : start + input_steps])
        y.append(trajectory[start + input_steps : start + total])
    return np.asarray(x), np.asarray(y)


def generate_parameter_split(
    output: Path,
    *,
    train_c: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0),
    test_c: tuple[float, ...] = (4.5, 5.0),
    q: float = -2.0,
    initial_states: tuple[tuple[float, float], ...] = (
        (-3.0, 0.5),
        (-2.7, -0.4),
        (-1.4, 0.7),
    ),
    step: float = 0.02,
    horizon: float = 25.0,
    input_steps: int = 50,
    forecast_steps: int = 25,
) -> None:
    """Create train/test files with test values of c unseen during training."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for split, c_values in {"train": train_c, "test": test_c}.items():
        xs, ys, params = [], [], []
        for c in c_values:
            p = LNETLMParameters(P=1.0, T=1.0, Q=q, c=c)
            for initial in initial_states:
                _, states = integrate(
                    lambda t, y: rhs(t, y, p),
                    np.asarray(initial),
                    (0.0, horizon),
                    step,
                    divergence_limit=1e5,
                )
                x, y = make_windows(states, input_steps, forecast_steps, stride=5)
                if len(x):
                    xs.append(x)
                    ys.append(y)
                    params.extend([asdict(p)] * len(x))

        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
        np.savez_compressed(output / f"{split}.npz", X=X, Y=Y)
        manifest[split] = {
            "samples": int(len(X)),
            "c_values": list(c_values),
            "parameter_records": len(params),
        }

    manifest["design"] = {
        "input_steps": input_steps,
        "forecast_steps": forecast_steps,
        "step": step,
        "split_principle": "hold out parameter values, not random points",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/generated"))
    args = parser.parse_args()
    generate_parameter_split(args.output)
