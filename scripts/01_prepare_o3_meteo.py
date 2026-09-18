from __future__ import annotations

import argparse
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import py7zr
except ImportError:  # pragma: no cover - handled at runtime
    py7zr = None


DEFAULT_STATIONS = [
    "前金",
    "前鎮",
    "小港",
    "左營",
    "楠梓",
    "仁武",
    "大寮",
    "鳳山",
    "林園",
    "美濃",
    "橋頭",
    "復興",
]

ITEMS = [
    "O3",
    "AMB_TEMP",
    "RH",
    "WIND_SPEED",
    "WIND_DIREC",
    "RAINFALL",
    "NO",
    "NO2",
    "NOx",
    "CO",
    "PM10",
    "PM2.5",
]

HOUR_COLUMNS = [f"{hour:02d}" for hour in range(24)]
INVALID_VALUE_PATTERN = re.compile(r"[#*xA]", flags=re.IGNORECASE)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hourly O3/meteo and daily MDA8 tables for Kaohsiung."
    )
    parser.add_argument("--air-dir", default="空氣資料", help="Directory with annual air-quality archives.")
    parser.add_argument("--sunshine-daily", default="outputs/tables/kaohsiung_sunshine_daily_2016_2025_extended.csv")
    parser.add_argument("--out-dir", default="outputs/tables")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--stations",
        nargs="*",
        default=DEFAULT_STATIONS,
        help="Kaohsiung station names to keep.",
    )
    return parser.parse_args()


def archive_year(path: Path) -> int | None:
    match = re.search(r"(20\d{2})", path.name)
    if not match:
        return None
    return int(match.group(1))


def find_archives(air_dir: Path, start_year: int, end_year: int) -> list[Path]:
    archives: list[Path] = []
    for path in sorted(air_dir.iterdir()):
        if path.suffix.lower() not in {".zip", ".7z"}:
            continue
        year = archive_year(path)
        if year is None or not (start_year <= year <= end_year):
            continue
        if path.name.startswith("全部_"):
            archives.append(path)
    return archives


def is_station_file(name: str, stations: Iterable[str]) -> bool:
    stem = Path(name).stem
    return any(station in stem for station in stations)


def clean_numeric(values: pd.Series) -> pd.Series:
    s = values.astype("string").str.strip()
    s = s.mask(s.str.upper().isin(["", "NA", "N/A", "NULL", "NAN", "NR", "ND"]))
    s = s.mask(s.str.upper().eq("T"), "0")
    s = s.mask(s.str.contains(INVALID_VALUE_PATTERN, na=False))
    return pd.to_numeric(s, errors="coerce")


def read_station_csv(handle, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(handle, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(col).strip() for col in df.columns]
    unnamed = [col for col in df.columns if col.startswith("Unnamed") or col == ""]
    if unnamed:
        df = df.drop(columns=unnamed)

    required = {"測站", "日期", "測項"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{source_name} missing columns: {sorted(missing)}")

    hour_cols = [col for col in HOUR_COLUMNS if col in df.columns]
    if len(hour_cols) != 24:
        raise ValueError(f"{source_name} has {len(hour_cols)} hour columns, expected 24")

    df["測項"] = df["測項"].astype("string").str.strip()
    df = df[df["測項"].isin(ITEMS)].copy()
    if df.empty:
        return pd.DataFrame()

    long = df.melt(
        id_vars=["測站", "日期", "測項"],
        value_vars=hour_cols,
        var_name="hour",
        value_name="value",
    )
    long["value"] = clean_numeric(long["value"])
    long["datetime"] = (
        pd.to_datetime(long["日期"], errors="coerce").dt.normalize()
        + pd.to_timedelta(long["hour"].astype(int), unit="h")
    )
    long = long.dropna(subset=["datetime"])

    wide = (
        long.pivot_table(
            index=["測站", "datetime"],
            columns="測項",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"測站": "site"})
    )
    wide.columns.name = None
    return wide


def iter_zip_station_frames(path: Path, stations: list[str]) -> Iterable[pd.DataFrame]:
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and is_station_file(name, stations)
        ]
        for name in names:
            print(f"  reading {name}", flush=True)
            with archive.open(name) as handle:
                yield read_station_csv(handle, name)


def iter_7z_station_frames(path: Path, stations: list[str]) -> Iterable[pd.DataFrame]:
    if py7zr is None:
        raise RuntimeError("py7zr is required for .7z archives. Install with: python -m pip install py7zr")

    with py7zr.SevenZipFile(path, mode="r") as archive:
        names = [
            name
            for name in archive.getnames()
            if name.lower().endswith(".csv") and is_station_file(name, stations)
        ]

    with tempfile.TemporaryDirectory(prefix="kaohsiung_o3_") as tmp:
        tmp_path = Path(tmp)
        with py7zr.SevenZipFile(path, mode="r") as archive:
            archive.extract(path=tmp_path, targets=names)

        for name in names:
            extracted = tmp_path / name
            print(f"  reading {name}", flush=True)
            yield read_station_csv(extracted, name)


def load_hourly(archives: list[Path], stations: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for archive in archives:
        print(f"Processing {archive.name}", flush=True)
        if archive.suffix.lower() == ".zip":
            iterator = iter_zip_station_frames(archive, stations)
        elif archive.suffix.lower() == ".7z":
            iterator = iter_7z_station_frames(archive, stations)
        else:
            continue

        for frame in iterator:
            if not frame.empty:
                frames.append(frame)

    if not frames:
        raise RuntimeError("No station data were loaded. Check archive names and station list.")

    hourly = pd.concat(frames, ignore_index=True, sort=False)
    hourly = hourly.drop_duplicates(subset=["site", "datetime"], keep="last")
    hourly = hourly.sort_values(["site", "datetime"]).reset_index(drop=True)

    for item in ITEMS:
        if item not in hourly.columns:
            hourly[item] = np.nan

    hourly = apply_physical_qc(hourly)
    hourly["date"] = hourly["datetime"].dt.normalize()
    hourly["year"] = hourly["datetime"].dt.year
    hourly["month"] = hourly["datetime"].dt.month
    hourly["hour"] = hourly["datetime"].dt.hour
    hourly["season"] = hourly["month"].map(month_to_season)
    return hourly


def apply_physical_qc(hourly: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly.copy()
    core = ["O3", "AMB_TEMP", "RH", "WIND_SPEED", "WIND_DIREC", "RAINFALL"]
    available_core = [col for col in core if col in hourly.columns]
    all_zero = hourly[available_core].fillna(0).eq(0).all(axis=1)
    hourly.loc[all_zero, available_core] = np.nan

    if "O3" in hourly.columns:
        hourly.loc[(hourly["O3"] <= 0) | (hourly["O3"] > 250), "O3"] = np.nan
    if "AMB_TEMP" in hourly.columns:
        hourly.loc[(hourly["AMB_TEMP"] < 5) | (hourly["AMB_TEMP"] > 45), "AMB_TEMP"] = np.nan
    if "RH" in hourly.columns:
        hourly.loc[(hourly["RH"] <= 0) | (hourly["RH"] > 100), "RH"] = np.nan
    if "WIND_SPEED" in hourly.columns:
        hourly.loc[(hourly["WIND_SPEED"] < 0) | (hourly["WIND_SPEED"] > 30), "WIND_SPEED"] = np.nan
    if "WIND_DIREC" in hourly.columns:
        hourly.loc[(hourly["WIND_DIREC"] < 0) | (hourly["WIND_DIREC"] > 360), "WIND_DIREC"] = np.nan
    if "RAINFALL" in hourly.columns:
        hourly.loc[(hourly["RAINFALL"] < 0) | (hourly["RAINFALL"] > 500), "RAINFALL"] = np.nan

    for pollutant in ["NO", "NO2", "NOx", "CO", "PM10", "PM2.5"]:
        if pollutant in hourly.columns:
            hourly.loc[hourly[pollutant] < 0, pollutant] = np.nan
    return hourly


def month_to_season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def compute_mda8(hourly: pd.DataFrame) -> pd.DataFrame:
    daily_frames: list[pd.DataFrame] = []
    for site, group in hourly.groupby("site", sort=True):
        o3 = group.set_index("datetime")["O3"].sort_index()
        if o3.empty:
            continue
        full_index = pd.date_range(o3.index.min(), o3.index.max(), freq="h")
        o3 = o3.reindex(full_index)
        rolling_mean = o3.rolling(window=8, min_periods=6).mean()
        rolling_count = o3.rolling(window=8, min_periods=1).count()
        tmp = pd.DataFrame(
            {
                "site": site,
                "date": rolling_mean.index.normalize(),
                "mda8_o3": rolling_mean.to_numpy(),
                "valid_8h_windows": (rolling_count.to_numpy() >= 6).astype(int),
            }
        )
        daily = (
            tmp.groupby(["site", "date"], as_index=False)
            .agg(mda8_o3=("mda8_o3", "max"), valid_8h_windows=("valid_8h_windows", "sum"))
        )
        daily_frames.append(daily)

    return pd.concat(daily_frames, ignore_index=True)


def add_daily_meteorology(hourly: pd.DataFrame, mda8: pd.DataFrame) -> pd.DataFrame:
    wind_rad = np.deg2rad(hourly["WIND_DIREC"])
    hourly = hourly.copy()
    hourly["wind_dir_sin_hourly"] = np.sin(wind_rad)
    hourly["wind_dir_cos_hourly"] = np.cos(wind_rad)

    daily_met = (
        hourly.groupby(["site", "date"], as_index=False)
        .agg(
            o3_hourly_count=("O3", "count"),
            temp_mean=("AMB_TEMP", "mean"),
            temp_max=("AMB_TEMP", "max"),
            rh_mean=("RH", "mean"),
            wind_speed_mean=("WIND_SPEED", "mean"),
            wind_dir_sin=("wind_dir_sin_hourly", "mean"),
            wind_dir_cos=("wind_dir_cos_hourly", "mean"),
            rain_sum=("RAINFALL", lambda x: x.sum(min_count=1)),
            no2_mean=("NO2", "mean"),
            nox_mean=("NOx", "mean"),
            co_mean=("CO", "mean"),
            pm25_mean=("PM2.5", "mean"),
        )
    )
    daily_met["wind_dir_mean"] = (
        np.degrees(np.arctan2(daily_met["wind_dir_sin"], daily_met["wind_dir_cos"])) + 360
    ) % 360

    daily = mda8.merge(daily_met, on=["site", "date"], how="left")
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day
    daily["doy"] = daily["date"].dt.dayofyear
    daily["season"] = daily["month"].map(month_to_season)
    return daily


def merge_sunshine(daily: pd.DataFrame, sunshine_path: Path) -> pd.DataFrame:
    if not sunshine_path.exists():
        print(f"Sunshine file not found: {sunshine_path}", flush=True)
        daily["sunshine_hours"] = np.nan
        return daily

    sunshine = pd.read_csv(sunshine_path, encoding="utf-8-sig")
    sunshine.columns = [str(col).strip() for col in sunshine.columns]
    if {"Date", "SunshineHours"}.issubset(sunshine.columns):
        sunshine = sunshine.rename(columns={"Date": "date", "SunshineHours": "sunshine_hours"})
    elif not {"date", "sunshine_hours"}.issubset(sunshine.columns):
        raise ValueError(f"{sunshine_path} must contain date/sunshine_hours or Date/SunshineHours columns")

    sunshine = sunshine.copy()
    sunshine["date"] = pd.to_datetime(sunshine["date"], errors="coerce").dt.normalize()
    sunshine["sunshine_hours"] = pd.to_numeric(sunshine["sunshine_hours"], errors="coerce")
    if "sunshine_missing" in sunshine.columns:
        sunshine["sunshine_missing"] = sunshine["sunshine_missing"].astype("string").str.lower().eq("true")
    elif "IsMissing" in sunshine.columns:
        sunshine["sunshine_missing"] = sunshine["IsMissing"].astype("string").str.lower().eq("true")
    else:
        sunshine["sunshine_missing"] = sunshine["sunshine_hours"].isna()

    return daily.merge(
        sunshine[["date", "sunshine_hours", "sunshine_missing"]],
        on="date",
        how="left",
    )


def write_outputs(hourly: pd.DataFrame, daily: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hourly_path = out_dir / "hourly_o3_meteo_2018_2025.csv"
    daily_path = out_dir / "daily_mda8_o3_2018_2025.csv"
    qc_path = out_dir / "data_availability_by_site_year.csv"

    hourly.to_csv(hourly_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    qc = (
        daily.groupby(["site", "year"], as_index=False)
        .agg(
            days=("date", "count"),
            days_with_mda8=("mda8_o3", "count"),
            mean_o3_hourly_count=("o3_hourly_count", "mean"),
            mean_valid_8h_windows=("valid_8h_windows", "mean"),
        )
    )
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {hourly_path}", flush=True)
    print(f"Wrote {daily_path}", flush=True)
    print(f"Wrote {qc_path}", flush=True)


def main() -> None:
    args = parse_args()
    air_dir = Path(args.air_dir)
    out_dir = Path(args.out_dir)
    archives = find_archives(air_dir, args.start_year, args.end_year)
    if not archives:
        raise RuntimeError(f"No annual archives found in {air_dir}")

    print("Archives:", ", ".join(path.name for path in archives), flush=True)
    hourly = load_hourly(archives, args.stations)
    mda8 = compute_mda8(hourly)
    daily = add_daily_meteorology(hourly, mda8)
    daily = merge_sunshine(daily, Path(args.sunshine_daily))
    write_outputs(hourly, daily, out_dir)


if __name__ == "__main__":
    main()
