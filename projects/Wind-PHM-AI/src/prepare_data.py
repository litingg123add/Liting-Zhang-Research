"""Download and compact the official EDP 2016 SCADA workbook."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd


SIGNALS_URL = "https://edp.com/sites/default/files/document/2025-04/Wind-Turbine-SCADA-signals-2016.xlsx"
FAILURES_URL = "https://edp.com/sites/default/files/document/2025-04/Historical-Failure-Logbook-2016.xlsx"

SELECTED_COLUMNS = [
    "Turbine_ID", "Timestamp", "Gen_RPM_Avg", "Rtr_RPM_Avg",
    "Amb_WindSpeed_Avg", "Amb_Temp_Avg", "Prod_LatestAvg_TotActPwr",
    "Grd_Prod_Pwr_Avg", "Blds_PitchAngle_Avg", "Nac_Temp_Avg",
    "Gen_Bear_Temp_Avg", "Gen_Bear2_Temp_Avg", "Gen_Phase1_Temp_Avg",
    "Gen_Phase2_Temp_Avg", "Gen_Phase3_Temp_Avg", "Hyd_Oil_Temp_Avg",
    "Gear_Oil_Temp_Avg", "Gear_Bear_Temp_Avg", "HVTrafo_Phase1_Temp_Avg",
    "HVTrafo_Phase2_Temp_Avg", "HVTrafo_Phase3_Temp_Avg",
]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    signal_file = args.raw_dir / "Wind-Turbine-SCADA-signals-2016.xlsx"
    failure_file = args.raw_dir / "Historical-Failure-Logbook-2016.xlsx"
    if not args.skip_download:
        download(SIGNALS_URL, signal_file)
        download(FAILURES_URL, failure_file)

    frame = pd.read_excel(signal_file, usecols=SELECTED_COLUMNS, engine="openpyxl")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True, errors="coerce")
    for column in SELECTED_COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    frame = frame.sort_values(["Turbine_ID", "Timestamp"]).reset_index(drop=True)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(args.processed_dir / "signals_2016_selected.pkl.gz", compression="gzip")
    pd.read_excel(failure_file).to_csv(args.processed_dir / "failures_2016.csv", index=False)


if __name__ == "__main__":
    main()
