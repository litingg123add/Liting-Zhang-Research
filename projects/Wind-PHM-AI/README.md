# Wind Turbine Predictive Health Monitoring

## 1. Problem

SCADA variables are strongly affected by operating conditions. A component temperature, for example, cannot be interpreted as healthy or unhealthy from one fixed threshold alone: the same temperature can correspond to different system states under different wind speeds, power levels and rotational speeds.

This project therefore focuses on **operating-condition-aware normal-behaviour modelling**. The main idea is:

> first estimate what a healthy turbine should look like under the current operating condition, then monitor the deviation from that expected behaviour.

This turns condition monitoring into a residual-analysis problem rather than a simple threshold problem.

## 2. Data

The current research is based on a public wind-turbine SCADA dataset with:
- multiple turbines,
- 10-minute operational measurements,
- environmental and operating variables,
- component temperatures,
- power / speed variables,
- maintenance and fault information.

Representative variables include wind speed, ambient temperature, wind direction, active power, rotor RPM, generator RPM, pitch angle, gearbox temperature, generator temperature and stator temperature.

Raw data are **not** committed to this repository. See `data/README.md`.

## 3. Research Pipeline

```
SCADA + maintenance records
        ↓
time-aware cleaning
        ↓
operating-condition features
        ↓
normal-behaviour model
        ↓
expected healthy response
        ↓
residual = observed - expected
        ↓
EWMA / robust anomaly score
        ↓
maintenance-event validation
```

### Step A — Preprocessing
- timestamp parsing and chronological sorting
- missing-value handling
- duplicate removal
- optional operating-range filtering
- no random shuffling before temporal evaluation

### Step B — Operating-condition representation
Examples include:
- wind speed and power
- rotor / generator speed
- ambient temperature
- pitch angle
- cyclic encoding of wind direction
- temperature differences relative to ambient conditions

### Step C — Normal-behaviour modelling
A regression model is trained on an earlier healthy period to estimate the expected value of a monitored variable under current operating conditions.

The implementation in `src/normal_behavior.py` uses XGBoost when available and falls back to scikit-learn's histogram gradient boosting model.

### Step D — Residual monitoring
For a target variable (y_t):

```
residual_t = observed_t - predicted_t
```

Residuals can then be transformed into:
- absolute deviation,
- EWMA-smoothed deviation,
- robust rolling z-score,
- persistence-based warning indicators.

### Step E — Event-oriented validation
The aim is not only to obtain a low regression error. The monitoring signal should also be evaluated against maintenance events using:
- chronological train / validation splits,
- pre-maintenance warning windows,
- false alarms outside event windows,
- lead time before maintenance,
- blind evaluation on periods not used during model fitting.

## 4. Repository Structure

```
Wind-PHM-AI/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── examples/
│   └── synthetic_demo.py
└── src/
    ├── preprocessing.py
    ├── feature_engineering.py
    ├── normal_behavior.py
    ├── monitoring.py
    └── evaluation.py
```

## 5. Quick Start

```bash
pip install -r requirements.txt
python examples/synthetic_demo.py
```

The synthetic demo creates a turbine-like time series, injects a gradual thermal degradation pattern, fits a normal-behaviour model on the earlier healthy period, and evaluates a residual-based warning signal.

## 6. Why This Project Matters

The core difficulty in industrial monitoring is that **normal behaviour is not stationary**. A reliable model must distinguish:
- operating-condition change,
- environmental change,
- sensor noise,
- actual degradation.

The project is therefore designed around generalization and interpretability rather than only classification accuracy.

## 7. Current Status

Active research prototype.

Current public code demonstrates the complete analysis logic without exposing restricted or oversized raw datasets. Dataset-specific column mapping and additional maintenance-event case studies are being consolidated for later release.
