# Technical Report: SCADA-Based Early Anomaly Detection for Wind Turbine T07

## 1. Research question

Can a normal-behaviour temperature model separate changes caused by wind and
operating load from abnormal generator-bearing thermal behaviour, and does the
resulting residual provide advance evidence before a confirmed EDP maintenance
event?

## 2. Exact case definition

| Required detail | Decision |
|---|---|
| Dataset | EDP Wind Farm 1 SCADA, 2016 |
| Turbine | T07 |
| Sampling | 10 minutes, UTC |
| Predicted signal | `Gen_Bear_Temp_Avg` |
| Ground-truth event | 2016-04-30 12:40 UTC |
| Failure-log component | `GENERATOR_BEARING` |
| Original remark | `High temperature in generator bearing (replaced sensor)` |
| Primary NBM | XGBoost gradient-boosted tree regressor |

## 3. Data and operating-regime definition

The 2016 official workbook has 207,905 rows, 83 SCADA fields and four turbines
(T01, T06, T07, T11). The model is evaluated only during power-producing
operation:

- grid power > 50 kW;
- generator speed > 200 rpm;
- wind speed between 3 and 25 m/s;
- complete target and feature values.

Inputs include wind speed, grid power, generator RPM, rotor RPM, blade pitch,
ambient temperature, nacelle temperature, trailing 1-hour and 6-hour operating
summaries, and cyclic hour-of-day terms. Target-temperature lags are excluded.

## 4. Temporal experimental design

| Window | Time range | Samples | Use |
|---|---|---:|---|
| Training | 2016-01-01 to 2016-01-31 | 2,212 | Fit NBM |
| Calibration | 2016-01-31 to 2016-03-01 | 3,183 | Residual centre, scale, interval, threshold |
| Nominal holdout | 2016-03-01 to 2016-03-31 | 2,129 | Accuracy and false-alarm evaluation |
| Pre-event monitoring | 2016-03-31 to 2016-04-30 | 2,386 | Lead-time evaluation only |

The windows are strictly chronological and non-overlapping.

## 5. Normal-behaviour model

XGBoost settings:
- 650 trees
- maximum depth 5
- learning rate 0.035
- subsample 0.85
- column-sample ratio 0.85
- minimum child weight 5
- L1 regularisation 0.10
- L2 regularisation 2.0
- random seed 42

### Chronological nominal-holdout performance

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---:|---:|---:|
| XGBoost NBM | **1.407** | **1.866** | **0.903** |
| Ridge baseline | 2.286 | 3.406 | 0.678 |

XGBoost reduced MAE by 38.46% relative to Ridge. The calibrated 95% residual
interval achieved 98.31% empirical coverage on the later nominal holdout.

## 6. Residual score and threshold

Residual:

`r(t) = observed generator-bearing temperature - NBM prediction`

Calibration residuals are converted to a robust one-sided score using the
calibration median and 1.4826 × MAD, followed by EWMA with alpha 0.20.

For the reproduced T07 run:
- calibration residual median: -0.092 °C;
- robust residual scale: 2.144 °C;
- EWMA threshold: 5.075 (calibration 99.5th percentile);
- persistence: >=12 threshold exceedances in a trailing 3-hour window;
- EWMA reset after operating-data gaps >30 minutes.

## 7. Detection result

The first persistent residual episode began at **2016-04-04 16:30 UTC**,
620.17 hours (**25.84 days**) before the logged event.

- nominal holdout false-alarm episodes: **0**
- nominal holdout alarm-sample rate: **0.00%**
- pre-event monitoring alarm-sample rate: **1.34%**
- logged-event detection within the predefined 30-day window: **yes**

The early evidence is intermittent rather than continuously active. The EDP
maintenance record states that a sensor was replaced, so the result should be
described as early abnormal thermal/sensor behaviour, not proof of mechanical
bearing damage.

## 8. Locked multi-event validation

With the protocol frozen and applied to four additional events:
- T06 generator replacement: detected 20.05 days before the event with zero
  nominal-holdout persistent episodes.
- T01 gearbox pump damage: detected 29.41 days before the event, but with 12
  nominal-holdout alarm episodes.
- T07 transformer overtemperature: missed.
- T11 hydraulic brake-circuit error: missed.

This is a useful limitation: the current method works well for some
generator-bearing cases but is not a general-purpose cross-component detector.
