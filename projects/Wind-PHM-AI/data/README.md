# Data

Raw SCADA and maintenance files are not committed to this repository.

The research pipeline is designed for a public wind-turbine SCADA dataset with 10-minute measurements from multiple turbines and associated maintenance / fault records.

Typical variables used by the current work include:
- wind speed
- ambient temperature
- wind direction
- active power
- rotor RPM
- generator RPM
- pitch angle
- gearbox / generator / stator temperatures
- maintenance-event timestamps

## Expected local layout

```
data/
├── raw/
│   ├── scada.csv
│   └── maintenance.csv
└── processed/
```

Large raw data, licensed data, and any industrial confidential data should stay outside Git.
