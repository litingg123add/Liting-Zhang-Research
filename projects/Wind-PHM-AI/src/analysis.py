"""Reproducible normal-behaviour modelling for the EDP wind-turbine data.

The implementation deliberately uses only information available before each
confirmed failure. It trains an XGBoost normal-behaviour model (NBM),
calibrates residuals on a later healthy window, and evaluates both predictive
accuracy and early-warning behaviour on strictly later time windows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


RANDOM_SEED = 42

BASE_FEATURES = [
    "Amb_WindSpeed_Avg",
    "Grd_Prod_Pwr_Avg",
    "Gen_RPM_Avg",
    "Rtr_RPM_Avg",
    "Blds_PitchAngle_Avg",
    "Amb_Temp_Avg",
    "Nac_Temp_Avg",
]

ROLLING_FEATURES = [
    "Amb_WindSpeed_Avg",
    "Grd_Prod_Pwr_Avg",
    "Gen_RPM_Avg",
    "Rtr_RPM_Avg",
    "Amb_Temp_Avg",
]


@dataclass(frozen=True)
class Case:
    key: str
    turbine: str
    target: str
    event_time: str
    component: str
    failure: str
    exclusion_intervals: tuple[tuple[str, str], ...] = ()


CASES = {
    "t01_gear_oil": Case(
        key="t01_gear_oil",
        turbine="T01",
        target="Gear_Oil_Temp_Avg",
        event_time="2016-07-18T02:10:00+00:00",
        component="Gearbox oil system",
        failure="Gearbox pump damaged",
    ),
    "t01_gear_bearing": Case(
        key="t01_gear_bearing",
        turbine="T01",
        target="Gear_Bear_Temp_Avg",
        event_time="2016-07-18T02:10:00+00:00",
        component="Gearbox bearing",
        failure="Gearbox pump damaged",
    ),
    "t06_generator_bearing": Case(
        key="t06_generator_bearing",
        turbine="T06",
        target="Gen_Bear_Temp_Avg",
        event_time="2016-07-11T19:48:00+00:00",
        component="Generator bearing",
        failure="Generator replaced",
        # A different-component hydraulic event was logged on 2016-04-04.
        # Embargoing +/-7 days prevents it contaminating the normal training set.
        exclusion_intervals=((
            "2016-03-28T18:53:00+00:00",
            "2016-04-11T18:53:00+00:00",
        ),),
    ),
    "t07_generator_bearing": Case(
        key="t07_generator_bearing",
        turbine="T07",
        target="Gen_Bear_Temp_Avg",
        event_time="2016-04-30T12:40:00+00:00",
        component="Generator bearing",
        failure="High temperature in generator bearing (sensor replaced)",
    ),
    "t07_transformer": Case(
        key="t07_transformer",
        turbine="T07",
        target="HVTrafo_Phase1_Temp_Avg",
        event_time="2016-07-10T03:46:00+00:00",
        component="Transformer phase 1",
        failure="High temperature transformer",
    ),
    "t11_hydraulic": Case(
        key="t11_hydraulic",
        turbine="T11",
        target="Hyd_Oil_Temp_Avg",
        event_time="2016-10-17T17:44:00+00:00",
        component="Hydraulic group",
        failure="Hydraulic group error in the brake circuit",
    ),
}


def load_signals(path: Path) -> pd.DataFrame:
    """Load the compact, typed cache produced from the official workbook."""
    frame = pd.read_pickle(path, compression="gzip")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
    return frame


def build_features(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Create causal operating-condition features using trailing windows only."""
    df = frame.sort_values("Timestamp").copy()
    for column in ROLLING_FEATURES:
        df[f"{column}_1h"] = df[column].rolling(6, min_periods=3).mean()
        df[f"{column}_6h"] = df[column].rolling(36, min_periods=12).mean()

    hour = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60.0
    df["Hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["Hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    feature_columns = BASE_FEATURES.copy()
    feature_columns += [f"{c}_1h" for c in ROLLING_FEATURES]
    feature_columns += [f"{c}_6h" for c in ROLLING_FEATURES]
    feature_columns += ["Hour_sin", "Hour_cos"]

    operating = (
        (df["Grd_Prod_Pwr_Avg"] > 50)
        & (df["Gen_RPM_Avg"] > 200)
        & df["Amb_WindSpeed_Avg"].between(3, 25)
        & df[target].notna()
    )
    df = df.loc[operating].dropna(subset=feature_columns + [target]).copy()
    return df, feature_columns


def metric_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_c": float(mean_absolute_error(y_true, y_pred)),
        "rmse_c": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def segmented_ewma(frame: pd.DataFrame, value: str, alpha: float = 0.20) -> pd.Series:
    """EWMA that resets after an operating-data gap longer than 30 minutes."""
    gap = frame["Timestamp"].diff().gt(pd.Timedelta(minutes=30)).cumsum()
    return frame.groupby(gap, sort=False)[value].transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean()
    )


def persistent_alarm(frame: pd.DataFrame, threshold: float) -> pd.Series:
    """Require >=12 exceedances in a trailing 3-hour window."""
    high = (frame["ewma_score"] > threshold).astype(int)
    indexed = pd.Series(high.to_numpy(), index=frame["Timestamp"])
    count = indexed.rolling("3h", min_periods=1).sum()
    return pd.Series((count >= 12).to_numpy(), index=frame.index)


def count_episodes(frame: pd.DataFrame, flag: str) -> int:
    if frame.empty:
        return 0
    current = frame[flag].astype(bool)
    new_episode = current & (
        ~current.shift(fill_value=False)
        | frame["Timestamp"].diff().gt(pd.Timedelta(minutes=30))
    )
    return int(new_episode.sum())


def analyse_case(
    signals: pd.DataFrame,
    case: Case,
    output_dir: Path,
    save_model: bool = False,
    make_plots: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, XGBRegressor]:
    event = pd.Timestamp(case.event_time)
    turbine = signals.loc[signals["Turbine_ID"] == case.turbine].copy()
    df, features = build_features(turbine, case.target)
    for start, end in case.exclusion_intervals:
        start_time, end_time = pd.Timestamp(start), pd.Timestamp(end)
        df = df.loc[~df["Timestamp"].between(start_time, end_time)].copy()

    train_end = event - pd.Timedelta(days=90)
    calibration_end = event - pd.Timedelta(days=60)
    healthy_test_end = event - pd.Timedelta(days=30)
    train = df.loc[df["Timestamp"] < train_end].copy()
    calibration = df.loc[
        (df["Timestamp"] >= train_end) & (df["Timestamp"] < calibration_end)
    ].copy()
    healthy_test = df.loc[
        (df["Timestamp"] >= calibration_end) & (df["Timestamp"] < healthy_test_end)
    ].copy()
    monitoring = df.loc[
        (df["Timestamp"] >= healthy_test_end) & (df["Timestamp"] <= event)
    ].copy()

    if min(len(train), len(calibration), len(healthy_test), len(monitoring)) < 250:
        raise ValueError(f"Insufficient operating data for {case.key}")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=650,
        max_depth=5,
        learning_rate=0.035,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=2.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(train[features], train[case.target])

    ridge = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
    )
    ridge.fit(train[features], train[case.target])

    evaluated = pd.concat(
        [
            calibration.assign(window="calibration"),
            healthy_test.assign(window="healthy_test"),
            monitoring.assign(window="monitoring"),
        ],
        ignore_index=True,
    ).sort_values("Timestamp")
    evaluated["prediction"] = model.predict(evaluated[features])
    evaluated["ridge_prediction"] = ridge.predict(evaluated[features])
    evaluated["residual"] = evaluated[case.target] - evaluated["prediction"]

    cal = evaluated["window"].eq("calibration")
    test = evaluated["window"].eq("healthy_test")
    mon = evaluated["window"].eq("monitoring")

    residual_center = float(evaluated.loc[cal, "residual"].median())
    mad = float((evaluated.loc[cal, "residual"] - residual_center).abs().median())
    residual_scale = max(1.4826 * mad, 0.05)

    evaluated["standardized_residual"] = (
        evaluated["residual"] - residual_center
    ) / residual_scale
    evaluated["positive_score"] = evaluated["standardized_residual"].clip(lower=0)
    evaluated["ewma_score"] = segmented_ewma(evaluated, "positive_score", alpha=0.20)

    threshold = float(evaluated.loc[cal, "ewma_score"].quantile(0.995))
    evaluated["persistent_alarm"] = persistent_alarm(evaluated, threshold)

    abs_cal_error = evaluated.loc[cal, "residual"].abs()
    conformal_q95 = float(abs_cal_error.quantile(0.95))
    evaluated["pi_lower"] = evaluated["prediction"] - conformal_q95
    evaluated["pi_upper"] = evaluated["prediction"] + conformal_q95

    first_alarm = evaluated.loc[mon & evaluated["persistent_alarm"], "Timestamp"].min()
    detected = bool(pd.notna(first_alarm))
    lead_hours = float((event - first_alarm).total_seconds() / 3600) if detected else None

    test_metrics = metric_dict(
        evaluated.loc[test, case.target], evaluated.loc[test, "prediction"].to_numpy()
    )
    ridge_metrics = metric_dict(
        evaluated.loc[test, case.target], evaluated.loc[test, "ridge_prediction"].to_numpy()
    )
    coverage = (
        (evaluated.loc[test, case.target] >= evaluated.loc[test, "pi_lower"])
        & (evaluated.loc[test, case.target] <= evaluated.loc[test, "pi_upper"])
    ).mean()

    result: dict[str, Any] = {
        "case": asdict(case),
        "windows": {
            "training_start": train["Timestamp"].min().isoformat(),
            "training_end": train["Timestamp"].max().isoformat(),
            "calibration_start": calibration["Timestamp"].min().isoformat(),
            "calibration_end": calibration["Timestamp"].max().isoformat(),
            "healthy_test_start": healthy_test["Timestamp"].min().isoformat(),
            "healthy_test_end": healthy_test["Timestamp"].max().isoformat(),
            "monitoring_start": monitoring["Timestamp"].min().isoformat(),
            "monitoring_end": monitoring["Timestamp"].max().isoformat(),
        },
        "sample_counts": {
            "training": int(len(train)),
            "calibration": int(len(calibration)),
            "healthy_test": int(len(healthy_test)),
            "monitoring": int(len(monitoring)),
        },
        "xgboost_test": test_metrics,
        "ridge_test": ridge_metrics,
        "xgboost_mae_improvement_vs_ridge_pct": float(
            100 * (ridge_metrics["mae_c"] - test_metrics["mae_c"]) / ridge_metrics["mae_c"]
        ),
        "prediction_interval": {
            "nominal_coverage": 0.95,
            "empirical_healthy_test_coverage": float(coverage),
            "half_width_c": conformal_q95,
        },
        "residual_monitor": {
            "calibration_median_c": residual_center,
            "calibration_robust_scale_c": residual_scale,
            "ewma_alpha": 0.20,
            "threshold_quantile": 0.995,
            "ewma_threshold": threshold,
            "persistence_rule": ">=12 exceedances in trailing 3 hours",
        },
        "detection": {
            "detected_in_30d_window": detected,
            "first_persistent_alarm": first_alarm.isoformat() if detected else None,
            "lead_time_hours": lead_hours,
            "lead_time_days": lead_hours / 24 if detected else None,
            "healthy_test_false_alarm_episodes": count_episodes(
                evaluated.loc[test], "persistent_alarm"
            ),
            "healthy_test_alarm_sample_rate": float(
                evaluated.loc[test, "persistent_alarm"].mean()
            ),
            "monitoring_alarm_sample_rate": float(
                evaluated.loc[mon, "persistent_alarm"].mean()
            ),
        },
        "feature_names": features,
    }

    if save_model or make_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        evaluated.to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")
    if save_model:
        model.get_booster().save_model(output_dir / "xgboost_nbm.json")
        joblib.dump({"ridge": ridge, "features": features}, output_dir / "baseline.joblib")
    if make_plots:
        create_figures(evaluated, model, case, result, features, output_dir)
    return result, evaluated, model


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def create_figures(
    evaluated: pd.DataFrame,
    model: XGBRegressor,
    case: Case,
    result: dict[str, Any],
    features: list[str],
    output_dir: Path,
) -> None:
    _style()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    event = pd.Timestamp(case.event_time)
    first_alarm_raw = result["detection"]["first_persistent_alarm"]
    first_alarm = pd.Timestamp(first_alarm_raw) if first_alarm_raw else None

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(evaluated["Timestamp"], evaluated[case.target], lw=0.8, color="#1f2937", label="Observed")
    ax.plot(evaluated["Timestamp"], evaluated["prediction"], lw=1.1, color="#2563eb", label="XGBoost NBM")
    ax.fill_between(
        evaluated["Timestamp"], evaluated["pi_lower"], evaluated["pi_upper"],
        color="#93c5fd", alpha=0.30, label="95% split-conformal interval"
    )
    ax.axvline(event, color="#dc2626", ls="--", lw=1.5, label="Confirmed failure")
    ax.set(title=f"{case.turbine}: operating-condition-aware temperature NBM", ylabel="Temperature (°C)", xlabel="UTC time")
    ax.legend(ncol=4, loc="upper left")
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.tight_layout()
    fig.savefig(figures / "01_temperature_prediction.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.3), sharex=True)
    axes[0].plot(evaluated["Timestamp"], evaluated["standardized_residual"], color="#475569", lw=0.75)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set(ylabel="Robust z-residual", title="Residual evolution before the confirmed failure")
    axes[1].plot(evaluated["Timestamp"], evaluated["ewma_score"], color="#7c3aed", lw=1.0, label="One-sided EWMA")
    axes[1].axhline(result["residual_monitor"]["ewma_threshold"], color="#f59e0b", ls="--", label="Calibrated threshold")
    for i, ax in enumerate(axes):
        ax.axvline(event, color="#dc2626", ls="--", lw=1.4, label="Confirmed event" if i == 1 else None)
        if first_alarm is not None:
            ax.axvline(first_alarm, color="#16a34a", ls=":", lw=1.5, label="First persistent excursion" if i == 1 else None)
    axes[1].set(ylabel="EWMA score", xlabel="UTC time")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figures / "02_residual_ewma.png", bbox_inches="tight")
    plt.close(fig)

    healthy = evaluated["window"].eq("healthy_test")
    y = evaluated.loc[healthy, case.target]
    p = evaluated.loc[healthy, "prediction"]
    lo = float(min(y.min(), p.min()))
    hi = float(max(y.max(), p.max()))
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(y, p, s=10, alpha=0.35, color="#2563eb", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color="#111827", ls="--", lw=1)
    met = result["xgboost_test"]
    ax.text(0.04, 0.96, f"MAE = {met['mae_c']:.2f} °C\nRMSE = {met['rmse_c']:.2f} °C\nR² = {met['r2']:.3f}", transform=ax.transAxes, va="top", bbox={"boxstyle":"round", "facecolor":"white", "alpha":0.9})
    ax.set(title="Chronological nominal-holdout performance", xlabel="Observed temperature (°C)", ylabel="Predicted temperature (°C)")
    fig.tight_layout()
    fig.savefig(figures / "03_healthy_test_scatter.png", bbox_inches="tight")
    plt.close(fig)

    importance = pd.Series(model.feature_importances_, index=features).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    importance.plot.barh(ax=ax, color="#0f766e")
    ax.set(title="XGBoost NBM feature importance (gain proxy)", xlabel="Normalised importance", ylabel="")
    fig.tight_layout()
    fig.savefig(figures / "04_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    mon = evaluated.loc[evaluated["window"].eq("monitoring")].copy()
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.1), sharex=True)
    axes[0].plot(mon["Timestamp"], mon["Grd_Prod_Pwr_Avg"], color="#0369a1", lw=0.75)
    axes[0].set(ylabel="Power (kW)", title="Operating context during the 30-day warning window")
    axes[1].plot(mon["Timestamp"], mon["Amb_WindSpeed_Avg"], color="#0891b2", lw=0.75)
    axes[1].set(ylabel="Wind (m/s)")
    axes[2].fill_between(mon["Timestamp"], 0, mon["persistent_alarm"].astype(int), step="mid", color="#dc2626", alpha=0.8)
    axes[2].set(ylabel="Alarm", xlabel="UTC time", yticks=[0, 1])
    for ax in axes:
        ax.axvline(event, color="#111827", ls="--", lw=1.2)
        if first_alarm is not None:
            ax.axvline(first_alarm, color="#16a34a", ls=":", lw=1.3)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.tight_layout()
    fig.savefig(figures / "05_operating_context_detection.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/signals_2016_selected.pkl.gz"))
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    signals = load_signals(args.data)
    case_keys = list(CASES) if args.case == "all" else [args.case]
    results = []
    for key in case_keys:
        case = CASES[key]
        result, _, _ = analyse_case(
            signals, case, args.output / key,
            save_model=args.save_model, make_plots=args.plots,
        )
        results.append(result)
        print(
            key,
            f"MAE={result['xgboost_test']['mae_c']:.3f}",
            f"R2={result['xgboost_test']['r2']:.3f}",
            f"detected={result['detection']['detected_in_30d_window']}",
            f"lead_d={result['detection']['lead_time_days']}",
            f"false_episodes={result['detection']['healthy_test_false_alarm_episodes']}",
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "screening_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    flat_rows = []
    for result in results:
        flat_rows.append(
            {
                "case": result["case"]["key"],
                "turbine": result["case"]["turbine"],
                "target": result["case"]["target"],
                "failure": result["case"]["failure"],
                **result["xgboost_test"],
                "lead_time_days": result["detection"]["lead_time_days"],
                "false_alarm_episodes": result["detection"]["healthy_test_false_alarm_episodes"],
                "healthy_alarm_sample_rate": result["detection"]["healthy_test_alarm_sample_rate"],
            }
        )
    pd.DataFrame(flat_rows).to_csv(args.output / "screening_summary.csv", index=False)


if __name__ == "__main__":
    main()
