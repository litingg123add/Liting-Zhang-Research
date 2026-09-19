# Data

Raw EDP SCADA data are not duplicated in this repository.

The analysis uses the public EDP Wind Farm 1 data for 2016. The project expects:
- wind-turbine SCADA signals;
- the historical failure logbook.

Primary dataset reference:
- https://doi.org/10.17632/zjxjnjp3xs.1

The repository also keeps the EDP source URLs in `src/prepare_data.py`. Because publisher / data-portal file paths can change over time, the DOI above is the stable reference.

## Preparation

When the direct EDP source files are reachable:

```bash
python src/prepare_data.py
```

If the source URLs have moved, download the two official 2016 files from the dataset source, place them under `data/raw/`, then run:

```bash
python src/prepare_data.py --skip-download
```

Expected generated files:

```text
data/raw/Wind-Turbine-SCADA-signals-2016.xlsx
data/raw/Historical-Failure-Logbook-2016.xlsx
data/processed/signals_2016_selected.pkl.gz
data/processed/failures_2016.csv
```
