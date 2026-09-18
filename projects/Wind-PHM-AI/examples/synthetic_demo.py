"""End-to-end synthetic demonstration of the Wind-PHM pipeline.

This example is intentionally synthetic: it demonstrates the code path without
redistributing the original research dataset.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from normal_behavior import NormalBehaviorModel
from monitoring import build_monitoring_signal, persistent_warning
from preprocessing import chronological_split


def make_synthetic_scada(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=n, freq="10min")

    wind = np.clip(rng.normal(8.5, 2.5, n), 0, 20)
    power = np.clip(120 * wind ** 2 + rng.normal(0, 250, n), 0, 5000)
    ambient = 12 + 8 * np.sin(np.linspace(0, 10 * np.pi, n)) + rng.normal(0, 1, n)
    rotor = np.clip(5 + 0.9 * wind + rng.normal(0, 0.6, n), 0, None)
    generator = 85 * rotor + rng.normal(0, 25, n)

    healthy_temp = (
        36
        + 0.0045 * power
        + 0.55 * ambient
        + 0.18 * wind
        + rng.normal(0, 1.0, n)
    )

    degradation = np.zeros(n)
    start = int(n * 0.82)
    degradation[start:] = np.linspace(0, 12, n - start)

    gearbox_temp = healthy_temp + degradation

    return pd.DataFrame(
        {
            "timestamp": t,
            "Wind Speed": wind,
            "Active Power": power,
            "Ambient Temperature": ambient,
            "Rotor RPM": rotor,
            "Generator RPM": generator,
            "Gearbox Temperature": gearbox_temp,
        }
    )


def main():
    df = make_synthetic_scada()

    features = [
        "Wind Speed",
        "Active Power",
        "Ambient Temperature",
        "Rotor RPM",
        "Generator RPM",
    ]
    target = "Gearbox Temperature"

    # Train only on the earlier chronological period.
    train, test = chronological_split(df, train_fraction=0.70)

    model = NormalBehaviorModel(features, target, prefer_xgboost=True)
    model.fit(train)

    report = model.evaluate(test.iloc[: int(len(test) * 0.35)])
    scored = model.residuals(df)
    signal = build_monitoring_signal(scored["residual"], ewma_span=24, z_window=144)
    warnings = persistent_warning(signal["anomaly_score"], threshold=3.0, min_consecutive=6)

    scored = pd.concat([scored, signal[["ewma_abs_residual", "anomaly_score"]]], axis=1)
    scored["warning"] = warnings

    print(f"Validation MAE:  {report.mae:.3f}")
    print(f"Validation RMSE: {report.rmse:.3f}")
    print(f"Warnings flagged: {int(scored['warning'].sum())}")
    print(scored.tail(10)[
        ["timestamp", target, "expected", "residual", "anomaly_score", "warning"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
