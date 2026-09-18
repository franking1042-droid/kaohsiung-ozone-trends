from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_FEATURES = [
    "temp_max",
    "rh_mean",
    "wind_speed_mean",
    "rain_log1p",
    "wind_dir_sin",
    "wind_dir_cos",
    "doy",
    "year_index",
]
PRECURSOR_FEATURES = ["nox_mean", "no2_mean", "co_mean", "pm25_mean"]
EVENT_FEATURES = ["temp_max", "rh_mean", "wind_speed_mean", "rain_sum", "sunshine_hours"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submission-readiness robustness checks for the Kaohsiung O3 study.")
    parser.add_argument("--daily", default="outputs/tables/daily_mda8_o3_2018_2025.csv")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--report", default="outputs/submission_readiness_report.md")
    parser.add_argument("--rf-trees", type=int, default=200)
    return parser.parse_args()


def load_trend_module() -> Any:
    trend_path = Path(__file__).with_name("07_trend_analysis.py")
    spec = importlib.util.spec_from_file_location("trend_analysis", trend_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load trend module: {trend_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def month_to_season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(month_to_season)
    df["rain_log1p"] = np.log1p(df["rain_sum"].clip(lower=0))
    df["year_index"] = df["year"] - df["year"].min()
    return df


def expected_days(year: int) -> int:
    return 366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365


def write_qc_tables(df: pd.DataFrame, tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (site, year), group in df.groupby(["site", "year"], sort=True):
        expected = expected_days(int(year))
        rows.append(
            {
                "site": site,
                "year": int(year),
                "expected_days": expected,
                "rows": len(group),
                "mda8_valid_days": group["mda8_o3"].notna().sum(),
                "mda8_availability": group["mda8_o3"].notna().sum() / expected,
                "mean_valid_8h_windows": group["valid_8h_windows"].mean(),
                "mean_o3_hourly_count": group["o3_hourly_count"].mean(),
                "temp_missing_rate": group["temp_max"].isna().mean(),
                "rh_missing_rate": group["rh_mean"].isna().mean(),
                "wind_speed_missing_rate": group["wind_speed_mean"].isna().mean(),
                "rain_missing_rate": group["rain_sum"].isna().mean(),
                "sunshine_missing_rate": group["sunshine_hours"].isna().mean(),
            }
        )
    station_year = pd.DataFrame(rows)
    station_year.to_csv(tables_dir / "submission_station_year_qc.csv", index=False, encoding="utf-8-sig")

    key_features = ["mda8_o3", "temp_max", "rh_mean", "wind_speed_mean", "rain_sum", "sunshine_hours"]
    summary = pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "n_sites": df["site"].nunique(),
                "start_date": df["date"].min().date(),
                "end_date": df["date"].max().date(),
                "min_station_year_mda8_availability": station_year["mda8_availability"].min(),
                "median_station_year_mda8_availability": station_year["mda8_availability"].median(),
                "station_years_below_90pct_mda8": int((station_year["mda8_availability"] < 0.90).sum()),
                **{f"{feature}_missing_rate": df[feature].isna().mean() for feature in key_features},
            }
        ]
    )
    summary.to_csv(tables_dir / "submission_qc_summary.csv", index=False, encoding="utf-8-sig")
    return station_year, summary


def write_cwa_validation(tables_dir: Path) -> pd.DataFrame:
    path = tables_dir / "cwa_air_station_daily_validation_2025.csv"
    if not path.exists():
        out = pd.DataFrame()
        out.to_csv(tables_dir / "submission_cwa_validation_metrics.csv", index=False, encoding="utf-8-sig")
        return out

    df = pd.read_csv(path, parse_dates=["date"])
    pairs = [
        ("temperature_mean", "air_station_temp_mean", "cwa_temp_mean"),
        ("temperature_max", "air_station_temp_max_mean", "cwa_temp_max"),
        ("relative_humidity", "air_station_rh_mean", "cwa_rh_mean"),
        ("wind_speed", "air_station_wind_speed_mean", "cwa_wind_speed_mean"),
        ("rainfall", "air_station_rain_sum_mean", "cwa_rain_sum"),
    ]
    rows = []
    for metric, air_col, cwa_col in pairs:
        valid = df[[air_col, cwa_col]].dropna()
        if valid.empty:
            corr = bias = mae = np.nan
            n = 0
        else:
            corr = valid[air_col].corr(valid[cwa_col])
            diff = valid[air_col] - valid[cwa_col]
            bias = diff.mean()
            mae = diff.abs().mean()
            n = len(valid)
        rows.append(
            {
                "metric": metric,
                "air_station_column": air_col,
                "cwa_column": cwa_col,
                "n_days": n,
                "pearson_r": corr,
                "mean_bias_air_minus_cwa": bias,
                "mae_air_minus_cwa": mae,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "submission_cwa_validation_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def monthly_metric(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    daily = (
        df.groupby("date", as_index=False)
        .agg(
            city_mean=("mda8_o3", "mean"),
            city_median=("mda8_o3", "median"),
            city_max=("mda8_o3", "max"),
            city_p90=("mda8_o3", lambda x: x.quantile(0.90)),
        )
        .dropna()
    )
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    monthly = (
        daily.groupby(["year", "month"], as_index=False)
        .agg(value=(metric_name, "mean"), n_days=(metric_name, "count"))
    )
    monthly["date"] = pd.to_datetime(monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-15")
    monthly["decimal_year"] = monthly["year"] + (monthly["month"] - 0.5) / 12.0
    return monthly


def write_trend_robustness(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    trend = load_trend_module()
    rows = []
    for start_year, end_year in [(2018, 2025), (2018, 2024), (2019, 2025)]:
        period = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
        for metric in ["city_mean", "city_median", "city_max", "city_p90"]:
            monthly = monthly_metric(period, metric)
            mk = trend.seasonal_mann_kendall(monthly, "value", "month", "decimal_year", alpha=0.05)
            slope = trend.sen_slope(monthly, "value", "decimal_year", strata_col="month")
            rows.append(
                {
                    "period": f"{start_year}-{end_year}",
                    "metric": metric,
                    "n_months": len(monthly),
                    "sen_slope_ppb_per_year": slope,
                    "kendall_tau": mk["kendall_tau"],
                    "p_value": mk["p_value"],
                    "direction": mk["direction"],
                    "trend": mk["trend"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "submission_trend_robustness.csv", index=False, encoding="utf-8-sig")
    return out


def city_event_table(df: pd.DataFrame) -> pd.DataFrame:
    city = (
        df.groupby("date", as_index=False)
        .agg(
            city_mean_mda8=("mda8_o3", "mean"),
            city_max_mda8=("mda8_o3", "max"),
            temp_max=("temp_max", "mean"),
            rh_mean=("rh_mean", "mean"),
            wind_speed_mean=("wind_speed_mean", "mean"),
            rain_sum=("rain_sum", "mean"),
            sunshine_hours=("sunshine_hours", "mean"),
        )
        .dropna(subset=["city_max_mda8"])
    )
    city["year"] = city["date"].dt.year
    city["month"] = city["date"].dt.month
    city["season"] = city["month"].map(month_to_season)
    return city


def write_event_robustness(df: pd.DataFrame, tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    city = city_event_table(df)
    definitions: list[tuple[str, float]] = [
        ("P90_city_max", city["city_max_mda8"].quantile(0.90)),
        ("P95_city_max", city["city_max_mda8"].quantile(0.95)),
        ("fixed_80ppb", 80.0),
        ("fixed_85ppb", 85.0),
    ]
    summary_rows = []
    season_frames = []
    for name, threshold in definitions:
        tmp = city.copy()
        tmp["is_event"] = tmp["city_max_mda8"] >= threshold
        events = tmp[tmp["is_event"]]
        row = {
            "definition": name,
            "threshold_ppb": threshold,
            "n_event_days": len(events),
            "event_fraction": len(events) / len(tmp),
            "event_city_max_mean": events["city_max_mda8"].mean(),
            "event_city_max_max": events["city_max_mda8"].max(),
        }
        for feature in EVENT_FEATURES:
            row[f"event_{feature}_median"] = events[feature].median()
            row[f"non_event_{feature}_median"] = tmp.loc[~tmp["is_event"], feature].median()
            row[f"event_minus_non_event_{feature}_median"] = (
                row[f"event_{feature}_median"] - row[f"non_event_{feature}_median"]
            )
        summary_rows.append(row)

        season_counts = (
            events.groupby("season", as_index=False)
            .size()
            .rename(columns={"size": "n_event_days"})
        )
        season_counts["definition"] = name
        season_counts["threshold_ppb"] = threshold
        season_frames.append(season_counts)

    summary = pd.DataFrame(summary_rows)
    by_season = pd.concat(season_frames, ignore_index=True)
    summary.to_csv(tables_dir / "submission_event_threshold_robustness.csv", index=False, encoding="utf-8-sig")
    by_season.to_csv(tables_dir / "submission_event_threshold_by_season.csv", index=False, encoding="utf-8-sig")
    return summary, by_season


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def prepare_model_subset(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    subset = df.dropna(subset=["mda8_o3", "site"] + features).copy()
    return subset


def fit_rf(train: pd.DataFrame, test: pd.DataFrame, features: list[str], n_estimators: int) -> dict[str, float]:
    rf_features = features + ["site"]
    x_train = pd.get_dummies(train[rf_features], columns=["site"], drop_first=False).astype(float)
    x_test = pd.get_dummies(test[rf_features], columns=["site"], drop_first=False)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0).astype(float)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        oob_score=False,
    )
    model.fit(x_train, train["mda8_o3"].to_numpy())
    pred = model.predict(x_test)
    return metrics(test["mda8_o3"].to_numpy(), pred)


def fit_gam(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict[str, float]:
    train = train.copy()
    test = test.copy()
    categories = pd.Categorical(train["site"])
    site_map = {site: code for code, site in enumerate(categories.categories)}
    train["site_code"] = train["site"].map(site_map).astype(float)
    test["site_code"] = test["site"].map(site_map).fillna(-1).astype(float)

    gam_features = features + ["site_code"]
    terms = s(0, n_splines=8)
    for idx in range(1, len(features)):
        terms += s(idx, n_splines=8)
    terms += f(len(features))

    model = LinearGAM(terms).fit(train[gam_features].to_numpy(), train["mda8_o3"].to_numpy())
    pred = model.predict(test[gam_features].to_numpy())
    return metrics(test["mda8_o3"].to_numpy(), pred)


def write_model_rolling_validation(df: pd.DataFrame, tables_dir: Path, rf_trees: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = [
        ("base", BASE_FEATURES),
        ("sunshine", BASE_FEATURES + ["sunshine_hours"]),
        ("precursors", BASE_FEATURES + PRECURSOR_FEATURES),
    ]
    rows = []
    for experiment, features in configs:
        subset = prepare_model_subset(df, features)
        for test_year in [2021, 2022, 2023, 2024, 2025]:
            train = subset[subset["year"] < test_year].copy()
            test = subset[subset["year"] == test_year].copy()
            if len(train) < 1000 or len(test) < 100:
                continue
            print(f"Rolling validation {experiment}: train < {test_year}, test {test_year}", flush=True)
            for model_name, func in [
                ("Random Forest", lambda tr, te, ft: fit_rf(tr, te, ft, rf_trees)),
                ("GAM", fit_gam),
            ]:
                result = func(train, test, features)
                rows.append(
                    {
                        "experiment": experiment,
                        "model": model_name,
                        "test_year": test_year,
                        "train_years": f"{train['year'].min()}-{train['year'].max()}",
                        "n_train": len(train),
                        "n_test": len(test),
                        **result,
                    }
                )
    performance = pd.DataFrame(rows)
    performance.to_csv(tables_dir / "submission_model_rolling_year_performance.csv", index=False, encoding="utf-8-sig")
    summary = (
        performance.groupby(["experiment", "model"], as_index=False)
        .agg(
            n_folds=("test_year", "count"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            min_r2=("r2", "min"),
            max_r2=("r2", "max"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
        )
        .sort_values(["experiment", "model"])
    )
    summary.to_csv(tables_dir / "submission_model_rolling_year_summary.csv", index=False, encoding="utf-8-sig")
    return performance, summary


def write_report(
    report_path: Path,
    qc_summary: pd.DataFrame,
    trend_robustness: pd.DataFrame,
    event_summary: pd.DataFrame,
    event_season: pd.DataFrame,
    model_summary: pd.DataFrame,
    cwa_validation: pd.DataFrame,
) -> None:
    def md_table(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except ImportError:
            return "```text\n" + frame.to_csv(index=False) + "```"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    qc = qc_summary.iloc[0]
    city_trend = trend_robustness[
        (trend_robustness["period"] == "2018-2025") & (trend_robustness["metric"] == "city_mean")
    ].iloc[0]
    lines = [
        "# Submission Readiness Report",
        "",
        "## Data QC",
        "",
        f"- Rows: {int(qc['n_rows'])}; sites: {int(qc['n_sites'])}; period: {qc['start_date']} to {qc['end_date']}.",
        f"- Minimum station-year MDA8 availability: {qc['min_station_year_mda8_availability']:.3f}.",
        f"- Station-years below 90% MDA8 availability: {int(qc['station_years_below_90pct_mda8'])}.",
        f"- Overall sunshine missing rate in daily table: {qc['sunshine_hours_missing_rate']:.4f}.",
        "",
        "## Trend Robustness",
        "",
        (
            "- Main citywide monthly mean Seasonal MK: "
            f"Sen slope {city_trend['sen_slope_ppb_per_year']:.3f} ppb/year, "
            f"p={city_trend['p_value']:.3f}, {city_trend['trend']}."
        ),
        "",
        md_table(trend_robustness),
        "",
        "## Event Threshold Robustness",
        "",
        md_table(
            event_summary[
                [
                    "definition",
                    "threshold_ppb",
                    "n_event_days",
                    "event_fraction",
                    "event_minus_non_event_rh_mean_median",
                    "event_minus_non_event_rain_sum_median",
                    "event_minus_non_event_sunshine_hours_median",
                ]
            ]
        ),
        "",
        "Season counts:",
        "",
        md_table(
            event_season.pivot_table(
                index="definition",
                columns="season",
                values="n_event_days",
                fill_value=0,
                aggfunc="sum",
            ).reset_index()
        ),
        "",
        "## Rolling-Year Model Validation",
        "",
        md_table(model_summary),
    ]
    if not cwa_validation.empty:
        lines.extend(["", "## CWA Cross-Validation", "", md_table(cwa_validation)])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}", flush=True)


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_daily(Path(args.daily))
    station_year_qc, qc_summary = write_qc_tables(df, tables_dir)
    print(f"QC rows: station-year={len(station_year_qc)}", flush=True)

    cwa_validation = write_cwa_validation(tables_dir)
    trend_robustness = write_trend_robustness(df, tables_dir)
    event_summary, event_season = write_event_robustness(df, tables_dir)
    model_performance, model_summary = write_model_rolling_validation(df, tables_dir, args.rf_trees)
    print(f"Rolling model rows: {len(model_performance)}", flush=True)

    write_report(
        Path(args.report),
        qc_summary,
        trend_robustness,
        event_summary,
        event_season,
        model_summary,
        cwa_validation,
    )


if __name__ == "__main__":
    main()
