from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import urllib3


MOENV_DOWNLOAD_URL = "https://data.moenv.gov.tw/api/frontstage/datastore/search-result.download"
CODIS_STATION_LIST_URL = "https://codis.cwa.gov.tw/api/station_list"
CODIS_STATION_URL = "https://codis.cwa.gov.tw/api/station"

MOENV_RESOURCES = {
    "AQX_P_07_station_basic_zh.csv": "fb92f773-27ca-470a-af04-6000397f7a4e",
    "AQX_P_07_station_basic_en.csv": "fe66f4bc-c2bf-4e32-b3b7-0fa780b1c986",
    "AQX_P_432_aqi_current_zh.csv": "8ff027dc-2da2-42e8-85de-78ac3faf470e",
    "GISEPA_P_03_station_location_map_zh.csv": "8231b829-3d53-4f9e-abfa-8444d359d5a8",
}

KAOHSIUNG_AQ_SITES = [
    "仁武",
    "前金",
    "前鎮",
    "大寮",
    "小港",
    "左營",
    "復興",
    "林園",
    "楠梓",
    "橋頭",
    "美濃",
    "鳳山",
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare supplementary data for the Kaohsiung O3 study.")
    parser.add_argument("--supp-dir", default="補充資料")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--start", default="2025-01-01T00:00:00")
    parser.add_argument("--end", default="2025-12-31T23:59:59")
    parser.add_argument("--cwa-station", default="467441", help="CWA Kaohsiung station after 2022-01-24.")
    return parser.parse_args()


def request_session() -> requests.Session:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    session = requests.Session()
    session.headers.update({"User-Agent": "kaohsiung-o3-research/1.0"})
    return session


def download_moenv_csv(session: requests.Session, rid: str) -> pd.DataFrame:
    payload = {
        "resource_id": rid,
        "limit": 5000,
        "offset": 0,
        "download_type": "csv",
    }
    response = session.post(MOENV_DOWNLOAD_URL, data=payload, timeout=90, verify=False)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig")


def save_moenv_resources(session: requests.Session, supp_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for filename, rid in MOENV_RESOURCES.items():
        df = download_moenv_csv(session, rid)
        frames[filename] = df
        df.to_csv(supp_dir / filename, index=False, encoding="utf-8-sig")
        print(f"Wrote {supp_dir / filename} rows={len(df)}", flush=True)
    return frames


def prepare_kaohsiung_station_metadata(frames: dict[str, pd.DataFrame], tables_dir: Path) -> pd.DataFrame:
    basic = frames["AQX_P_07_station_basic_zh.csv"].copy()
    aqi = frames["AQX_P_432_aqi_current_zh.csv"].copy()

    basic["siteid"] = basic["siteid"].astype(str)
    aqi["siteid"] = aqi["siteid"].astype(str)

    basic = basic[basic["sitename"].isin(KAOHSIUNG_AQ_SITES)].copy()
    aqi = aqi[["siteid", "sitename", "longitude", "latitude", "publishtime"]].rename(
        columns={
            "longitude": "aqi_longitude",
            "latitude": "aqi_latitude",
            "publishtime": "aqi_publishtime",
        }
    )
    merged = basic.merge(aqi, on=["siteid", "sitename"], how="left")
    merged = merged.rename(
        columns={
            "sitename": "site",
            "siteengname": "site_eng_name",
            "areaname": "air_quality_zone",
            "county": "county",
            "township": "township",
            "siteaddress": "site_address",
            "twd97lon": "longitude",
            "twd97lat": "latitude",
            "sitetype": "site_type",
            "siteid": "site_id",
        }
    )
    ordered = [
        "site",
        "site_id",
        "site_eng_name",
        "site_type",
        "air_quality_zone",
        "county",
        "township",
        "site_address",
        "longitude",
        "latitude",
        "aqi_longitude",
        "aqi_latitude",
        "aqi_publishtime",
    ]
    merged = merged[ordered].sort_values("site_id").reset_index(drop=True)
    out = tables_dir / "kaohsiung_air_quality_station_metadata.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {out} rows={len(merged)}", flush=True)
    return merged


def get_nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return np.nan
        value = value.get(key)
    if value in ("", "--"):
        return np.nan
    return value


def download_codis_station_list(session: requests.Session, supp_dir: Path) -> pd.DataFrame:
    response = session.get(CODIS_STATION_LIST_URL, timeout=90, verify=False)
    response.raise_for_status()
    payload = response.json()
    rows: list[dict[str, Any]] = []

    def walk(obj: Any, station_type: str | None = None) -> None:
        if isinstance(obj, dict):
            if "stationID" in obj and "stationName" in obj:
                rec = obj.copy()
                rec["codis_station_type_key"] = station_type
                rows.append(rec)
            for key, value in obj.items():
                next_type = key if key in {"cwb", "auto_C1", "auto_C0", "agr", "autotypeA"} else station_type
                walk(value, next_type)
        elif isinstance(obj, list):
            for value in obj:
                walk(value, station_type)

    walk(payload)
    stations = pd.DataFrame(rows).drop_duplicates(subset=["stationID"], keep="first")
    stations.to_csv(supp_dir / "CODiS_station_list.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {supp_dir / 'CODiS_station_list.csv'} rows={len(stations)}", flush=True)
    return stations


def download_codis_daily(session: requests.Session, station_id: str, start: str, end: str) -> dict[str, Any]:
    payload = {
        "type": "report_month",
        "stn_type": "cwb",
        "stn_ID": station_id,
        "start": start,
        "end": end,
        "more": "",
    }
    response = session.post(CODIS_STATION_URL, data=payload, timeout=180, verify=False)
    response.raise_for_status()
    return response.json()


def parse_codis_daily(payload: dict[str, Any], station_id: str) -> pd.DataFrame:
    data = payload.get("data") or []
    if not data:
        return pd.DataFrame()
    dts = data[0].get("dts") or []
    rows: list[dict[str, Any]] = []
    for row in dts:
        rows.append(
            {
                "station_id": station_id,
                "date": pd.to_datetime(row.get("DataDate"), errors="coerce").date(),
                "cwa_temp_mean": get_nested(row, "AirTemperature", "Mean"),
                "cwa_temp_max": get_nested(row, "AirTemperature", "Maximum"),
                "cwa_temp_min": get_nested(row, "AirTemperature", "Minimum"),
                "cwa_rh_mean": get_nested(row, "RelativeHumidity", "Mean"),
                "cwa_rh_min": get_nested(row, "RelativeHumidity", "Minimum"),
                "cwa_wind_speed_mean": get_nested(row, "WindSpeed", "Mean"),
                "cwa_wind_dir_prevailing": get_nested(row, "WindDirection", "Prevailing"),
                "cwa_peak_gust": get_nested(row, "PeakGust", "Maximum"),
                "cwa_rain_sum_raw": get_nested(row, "Precipitation", "Accumulation"),
                "cwa_precip_duration": get_nested(row, "PrecipitationDuration", "Total"),
                "cwa_sunshine_hours": get_nested(row, "SunshineDuration", "Total"),
                "cwa_sunshine_rate": get_nested(row, "SunshineDuration", "Rate"),
                "cwa_global_solar_radiation_mj_m2": get_nested(row, "GlobalSolarRadiation", "Accumulation"),
                "cwa_global_solar_hourly_max": get_nested(row, "GlobalSolarRadiation", "HourlyMaximum"),
                "cwa_visibility": get_nested(row, "Visibility", "Mean"),
                "cwa_total_cloud_amount": get_nested(row, "TotalCloudAmount", "Mean"),
                "temp_mean_flag": get_nested(row, "AirTemperature", "Meanf"),
                "rh_mean_flag": get_nested(row, "RelativeHumidity", "Meanf"),
                "rain_sum_flag": get_nested(row, "Precipitation", "Accumulationf"),
                "sunshine_flag": get_nested(row, "SunshineDuration", "Totalf"),
                "solar_flag": get_nested(row, "GlobalSolarRadiation", "Accumulationf"),
            }
        )
    daily = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in daily.columns:
        if col not in {"station_id", "date"} and not col.endswith("_flag"):
            daily[col] = pd.to_numeric(daily[col], errors="coerce")
    if "cwa_rain_sum_raw" in daily.columns:
        rain_insert_at = daily.columns.get_loc("cwa_rain_sum_raw") + 1
        rain_clean = daily["cwa_rain_sum_raw"].mask(daily["cwa_rain_sum_raw"] < 0, 0.0)
        daily.insert(rain_insert_at, "cwa_rain_sum", rain_clean)
    return daily


def save_codis_daily(
    session: requests.Session,
    supp_dir: Path,
    tables_dir: Path,
    station_id: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    raw = download_codis_daily(session, station_id, start, end)
    raw_path = supp_dir / f"CODiS_{station_id}_daily_2025_raw.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    daily = parse_codis_daily(raw, station_id)
    supp_csv = supp_dir / f"CODiS_{station_id}_daily_2025.csv"
    out_csv = tables_dir / "cwa_kaohsiung_daily_meteorology_2025.csv"
    daily.to_csv(supp_csv, index=False, encoding="utf-8-sig")
    daily.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {supp_csv} rows={len(daily)}", flush=True)
    print(f"Wrote {out_csv} rows={len(daily)}", flush=True)
    return daily


def write_extended_sunshine(daily: pd.DataFrame, tables_dir: Path, supp_dir: Path) -> None:
    existing_path = Path("高雄日照時數_2016-2024_逐日.csv")
    if existing_path.exists():
        existing = pd.read_csv(existing_path, encoding="utf-8-sig")
        existing = existing.rename(columns={"Date": "date", "SunshineHours": "sunshine_hours"})
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        existing = existing[["date", "sunshine_hours"]].copy()
        existing["source_station_id"] = "467440"
        existing["source"] = "existing_CODiS_export"
    else:
        existing = pd.DataFrame(columns=["date", "sunshine_hours", "source_station_id", "source"])

    add = daily[["date", "cwa_sunshine_hours"]].rename(columns={"cwa_sunshine_hours": "sunshine_hours"}).copy()
    add["source_station_id"] = daily["station_id"]
    add["source"] = "CODiS_api"
    combined = pd.concat([existing, add], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    combined["year"] = pd.to_datetime(combined["date"]).dt.year
    combined["month"] = pd.to_datetime(combined["date"]).dt.month
    combined["day"] = pd.to_datetime(combined["date"]).dt.day

    out = tables_dir / "kaohsiung_sunshine_daily_2016_2025_extended.csv"
    supp_out = supp_dir / "kaohsiung_sunshine_daily_2016_2025_extended.csv"
    combined.to_csv(out, index=False, encoding="utf-8-sig")
    combined.to_csv(supp_out, index=False, encoding="utf-8-sig")
    print(f"Wrote {out} rows={len(combined)}", flush=True)


def write_cwa_air_station_validation(daily_cwa: pd.DataFrame, tables_dir: Path) -> None:
    daily_path = tables_dir / "daily_mda8_o3_2018_2025.csv"
    if not daily_path.exists():
        print(f"Daily O3 table not found, skipped CWA validation merge: {daily_path}", flush=True)
        return

    air = pd.read_csv(daily_path, parse_dates=["date"])
    air_2025 = air[air["date"].dt.year == 2025].copy()
    if air_2025.empty:
        print("No 2025 rows in daily O3 table, skipped CWA validation merge.", flush=True)
        return

    city = (
        air_2025.groupby("date", as_index=False)
        .agg(
            city_mean_mda8_o3=("mda8_o3", "mean"),
            city_max_mda8_o3=("mda8_o3", "max"),
            n_o3_sites=("mda8_o3", "count"),
            air_station_temp_mean=("temp_mean", "mean"),
            air_station_temp_max_mean=("temp_max", "mean"),
            air_station_rh_mean=("rh_mean", "mean"),
            air_station_wind_speed_mean=("wind_speed_mean", "mean"),
            air_station_rain_sum_mean=("rain_sum", "mean"),
        )
    )

    cwa = daily_cwa.copy()
    cwa["date"] = pd.to_datetime(cwa["date"])
    merged = city.merge(cwa, on="date", how="left")
    out = tables_dir / "cwa_air_station_daily_validation_2025.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {out} rows={len(merged)}", flush=True)


def main() -> None:
    args = parse_args()
    supp_dir = Path(args.supp_dir)
    tables_dir = Path(args.tables_dir)
    supp_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    session = request_session()
    frames = save_moenv_resources(session, supp_dir)
    prepare_kaohsiung_station_metadata(frames, tables_dir)
    stations = download_codis_station_list(session, supp_dir)
    stations[stations["stationID"].isin(["467440", args.cwa_station])].to_csv(
        tables_dir / "cwa_kaohsiung_station_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily = save_codis_daily(session, supp_dir, tables_dir, args.cwa_station, args.start, args.end)
    write_extended_sunshine(daily, tables_dir, supp_dir)
    write_cwa_air_station_validation(daily, tables_dir)


if __name__ == "__main__":
    main()
