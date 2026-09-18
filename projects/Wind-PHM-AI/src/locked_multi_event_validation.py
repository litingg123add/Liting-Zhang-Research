"""Locked-protocol multi-event validation.

This script intentionally imports and reuses analyse_case without changing
its model, feature, temporal-split, residual, EWMA, threshold-selection, or
persistence logic. Event-to-target mappings are fixed below before execution.
The primary T07 case is rerun as a development reference; the four remaining
cases are independent locked-protocol validation events.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import CASES, analyse_case, count_episodes, load_signals


PRIMARY_REFERENCE = "t07_generator_bearing"

LOCKED_VALIDATION_CASES = (
    "t01_gear_oil",
    "t06_generator_bearing",
    "t07_transformer",
    "t11_hydraulic",
)

EXPECTED_PROTOCOL = {
    "model": "XGBRegressor; parameters defined in src/analysis.py",
    "operating_filter": "power > 50 kW; generator RPM > 200; wind 3-25 m/s",
    "temporal_windows": "train < event-90d; calibration -90d:-60d; nominal holdout -60d:-30d; monitoring -30d:event",
    "residual": "observed target temperature - XGBoost NBM prediction",
    "standardisation": "calibration median and 1.4826*MAD; one-sided positive residual",
    "ewma_alpha": 0.20,
    "threshold_selection": "99.5th percentile of each event's calibration-window EWMA",
    "persistence": ">=12 exceedances in trailing 3 hours",
    "gap_reset": "EWMA reset after operating-data gap >30 minutes",
}


def status_label(detected: bool, false_episodes: int) -> str:
    if not detected:
        return "Missed"
    if false_episodes:
        return "Detected, but false alarms present"
    return "Detected with zero holdout episodes"


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    output = project / "results" / "locked_multi_event_validation"
    output.mkdir(parents=True, exist_ok=True)

    analysis_path = project / "src" / "analysis.py"
    protocol_manifest = {
        "analysis_py_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "primary_reference": PRIMARY_REFERENCE,
        "locked_validation_cases": list(LOCKED_VALIDATION_CASES),
        "protocol": EXPECTED_PROTOCOL,
        "event_target_mapping": {
            key: {
                "turbine": CASES[key].turbine,
                "event_time": CASES[key].event_time,
                "failure_log_remark": CASES[key].failure,
                "target": CASES[key].target,
                "predeclared_exclusion_intervals": list(CASES[key].exclusion_intervals),
            }
            for key in (PRIMARY_REFERENCE, *LOCKED_VALIDATION_CASES)
        },
    }
    with (output / "locked_protocol_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol_manifest, handle, indent=2, ensure_ascii=False)

    signals = load_signals(project / "data" / "processed" / "signals_2016_selected.pkl.gz")
    rows: list[dict[str, object]] = []
    raw_results: list[dict[str, object]] = []

    for key in (PRIMARY_REFERENCE, *LOCKED_VALIDATION_CASES):
        case = CASES[key]
        result, evaluated, _ = analyse_case(
            signals,
            case,
            output / "cases" / key,
            save_model=True,
            make_plots=False,
        )
        raw_results.append(result)

        monitor = evaluated.loc[evaluated["window"].eq("monitoring")]
        monitor_episodes = count_episodes(monitor, "persistent_alarm")
        detection = result["detection"]
        false_episodes = int(detection["healthy_test_false_alarm_episodes"])
        detected = bool(detection["detected_in_30d_window"])

        assert result["residual_monitor"]["ewma_alpha"] == EXPECTED_PROTOCOL["ewma_alpha"]
        assert result["residual_monitor"]["threshold_quantile"] == 0.995
        assert result["residual_monitor"]["persistence_rule"] == EXPECTED_PROTOCOL["persistence"]

        rows.append(
            {
                "role": "development_reference" if key == PRIMARY_REFERENCE else "locked_validation",
                "case": key,
                "turbine": case.turbine,
                "component": case.component,
                "target": case.target,
                "event_time": case.event_time,
                "failure_log_remark": case.failure,
                "mae_c": result["xgboost_test"]["mae_c"],
                "rmse_c": result["xgboost_test"]["rmse_c"],
                "r2": result["xgboost_test"]["r2"],
                "ewma_threshold": result["residual_monitor"]["ewma_threshold"],
                "detected_in_30d_window": detected,
                "first_persistent_alarm": detection["first_persistent_alarm"],
                "lead_time_days": detection["lead_time_days"],
                "nominal_holdout_false_alarm_episodes": false_episodes,
                "nominal_holdout_alarm_sample_rate": detection["healthy_test_alarm_sample_rate"],
                "monitoring_alarm_episodes": monitor_episodes,
                "monitoring_alarm_sample_rate": detection["monitoring_alarm_sample_rate"],
                "status": status_label(detected, false_episodes),
                "training_samples": result["sample_counts"]["training"],
                "calibration_samples": result["sample_counts"]["calibration"],
                "nominal_holdout_samples": result["sample_counts"]["healthy_test"],
                "monitoring_samples": result["sample_counts"]["monitoring"],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output / "locked_multi_event_summary.csv", index=False)
    with (output / "locked_multi_event_results.json").open("w", encoding="utf-8") as handle:
        json.dump(raw_results, handle, indent=2, ensure_ascii=False)

    validation = summary.loc[summary["role"].eq("locked_validation")]
    aggregate = {
        "number_of_additional_events": int(len(validation)),
        "detected_events_regardless_of_false_alarms": int(validation["detected_in_30d_window"].sum()),
        "detected_with_zero_holdout_false_alarm_episodes": int(
            (validation["detected_in_30d_window"] & validation["nominal_holdout_false_alarm_episodes"].eq(0)).sum()
        ),
        "missed_events": int((~validation["detected_in_30d_window"]).sum()),
        "events_with_holdout_false_alarm_episodes": int(
            validation["nominal_holdout_false_alarm_episodes"].gt(0).sum()
        ),
        "clean_event_detection_rate": float(
            (validation["detected_in_30d_window"] & validation["nominal_holdout_false_alarm_episodes"].eq(0)).mean()
        ),
    }
    with (output / "aggregate_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    make_comparison_figure(summary, output / "locked_multi_event_comparison.png")
    print(summary.to_string(index=False))
    print("\nAGGREGATE")
    print(json.dumps(aggregate, indent=2))


def make_comparison_figure(summary: pd.DataFrame, output: Path) -> None:
    labels = [f"{r.turbine} — {r.component}" for r in summary.itertuples()]
    lead = summary["lead_time_days"].fillna(0).to_numpy(dtype=float)
    false = summary["nominal_holdout_false_alarm_episodes"].to_numpy(dtype=int)
    detected = summary["detected_in_30d_window"].to_numpy(dtype=bool)
    y = np.arange(len(summary))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), gridspec_kw={"width_ratios": [1.65, 1]})

    bars = axes[0].barh(y, lead, color="#3b82f6", alpha=0.82)
    for i, bar in enumerate(bars):
        if not detected[i]:
            bar.set_color("#cbd5e1")
            axes[0].text(0.35, i, "Missed", va="center", ha="left", color="#475569", fontweight="bold")
        else:
            axes[0].text(lead[i] + 0.35, i, f"{lead[i]:.1f} d", va="center", ha="left", color="#1e3a5f")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Lead time from first persistent alarm (days)")
    axes[0].set_title("Locked-protocol event detection")
    axes[0].set_xlim(0, max(31, float(lead.max()) + 4))

    axes[1].barh(y, false, color="#f59e0b", alpha=0.85)
    for i, value in enumerate(false):
        axes[1].text(value + 0.18, i, str(value), va="center", ha="left", color="#78350f")
    axes[1].set_yticks(y, ["" for _ in y])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Persistent episodes in nominal holdout")
    axes[1].set_title("False-alarm episodes")
    axes[1].set_xlim(0, max(13, int(false.max()) + 2))

    fig.suptitle("Primary reference plus four independent maintenance events", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
