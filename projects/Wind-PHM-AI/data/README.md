# Data

Raw EDP SCADA data are not duplicated in this repository.

Run:

```bash
python src/prepare_data.py
```

The script downloads the official 2016 EDP SCADA workbook and maintenance log,
then creates a compact local cache used by the analysis.

Official source:
- https://edp.com/en/innovation/data
- https://doi.org/10.17632/zjxjnjp3xs.1

Expected generated files:

```text
data/raw/Wind-Turbine-SCADA-signals-2016.xlsx
data/raw/Historical-Failure-Logbook-2016.xlsx
data/processed/signals_2016_selected.pkl.gz
data/processed/failures_2016.csv
```
