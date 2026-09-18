# Operating-Condition-Aware Wind-Turbine Condition Monitoring

This repository contains a reproducible normal-behaviour modelling (NBM) case
study using the public EDP wind-farm SCADA data. The primary analysis predicts
the generator-bearing temperature of turbine T07 from operating and ambient
conditions, then monitors robust temperature residuals against a confirmed
maintenance log.

## Headline result

| Item | Reproduced result |
|---|---|
| Turbine | T07 |
| Target | `Gen_Bear_Temp_Avg` |
| Logged event | 2016-04-30 12:40 UTC — high generator-bearing temperature; sensor replaced |
| NBM | XGBoost tree regressor with causal 1 h / 6 h operating summaries |
| Chronological holdout | MAE 1.41 °C; RMSE 1.87 °C; R² 0.903 |
| Residual monitor | robust one-sided z-residual + EWMA (alpha 0.20) |
| Alarm threshold | 99.5th percentile of calibration EWMA; 12 exceedances in 3 h |
| First persistent excursion | **25.84 days (620.2 h)** before the logged event |
| Nominal holdout false alarms | **0 persistent episodes** in a separate 30-day window |

The lead time refers to the first intermittent persistent residual excursion,
not to an alarm that remained active continuously for 25.84 days. The EDP log
documents a sensor replacement; this project therefore does **not** claim a
mechanical bearing failure or remaining-useful-life prediction.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/prepare_data.py
python src/analysis.py \
  --case t07_generator_bearing \
  --data data/processed/signals_2016_selected.pkl.gz \
  --output results/final \
  --save-model --plots
```

`prepare_data.py` downloads the official 2016 EDP SCADA workbook and failure
log, converts numeric columns, orders records by turbine and UTC timestamp, and
creates a compact cache. The raw EDP workbook contains 207,905 ten-minute
records and 83 fields for T01, T06, T07, and T11.

## Leakage controls

- All splits are chronological and non-overlapping.
- The model does not use the target temperature or any future value as a feature.
- Rolling features are trailing (causal) windows only.
- Threshold selection uses the calibration window only.
- Accuracy and false alarms are measured on a later, untouched nominal holdout.
- The final 30 days before the logged event are used only for monitoring.

## Repository map

```text
src/prepare_data.py       Download, type conversion, compact cache
src/analysis.py           Feature engineering, NBM, residual monitor, figures
src/locked_multi_event_validation.py  Frozen-protocol validation runner
results/t07_generator_bearing/metrics.json  Reproduced primary-case metrics
results/locked_multi_event_summary.csv      Multi-event validation summary
TECHNICAL_REPORT.md       Methods, exact results, interpretation, limitations
```

## Locked multi-event validation

The original method and threshold-selection protocol were subsequently frozen
and applied to four additional maintenance events. One event (T06 generator
replacement) was detected 20.05 days in advance with zero persistent episodes
in its nominal holdout. T01 was detected but had 12 holdout alarm episodes;
the T07 transformer and T11 hydraulic cases were missed.

These results are deliberately retained because they show both the useful cases
and the limitations of the current method rather than only reporting a
successful example.

## Data provenance

- EDP official data portal: https://edp.com/en/innovation/data
- Dataset DOI: https://doi.org/10.17632/zjxjnjp3xs.1

The EDP files are public research data. Raw data are not duplicated in this
repository; `src/prepare_data.py` downloads the official files.
