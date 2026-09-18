from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "revision"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_prepare_module():
    path = ROOT / "scripts" / "01_prepare_o3_meteo.py"
    spec = importlib.util.spec_from_file_location("prepare_o3_meteo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
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


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def count_rule(rows: list[dict], rule: str, variable: str, checked: int, deleted: int, note: str = "") -> None:
    rows.append(
        {
            "task": "T6",
            "rule": rule,
            "variable": variable,
            "n_checked": int(checked),
            "n_deleted": int(deleted),
            "deleted_pct_of_checked": float(deleted / checked * 100) if checked else np.nan,
            "note": note,
        }
    )


def audit_physical_qc() -> tuple[pd.DataFrame, pd.DataFrame]:
    prep = load_prepare_module()
    archives = prep.find_archives(ROOT / "空氣資料", 2018, 2025)
    rows: list[dict] = []
    wind_gt30_records: list[pd.DataFrame] = []

    for archive in archives:
        print(f"[T6] Auditing {archive.name}", flush=True)
        if archive.suffix.lower() == ".zip":
            iterator = prep.iter_zip_station_frames(archive, prep.DEFAULT_STATIONS)
        elif archive.suffix.lower() == ".7z":
            iterator = prep.iter_7z_station_frames(archive, prep.DEFAULT_STATIONS)
        else:
            continue

        for frame in iterator:
            if frame.empty:
                continue
            frame = frame.copy()
            source = archive.name
            year = prep.archive_year(archive)

            core = ["O3", "AMB_TEMP", "RH", "WIND_SPEED", "WIND_DIREC", "RAINFALL"]
            available_core = [col for col in core if col in frame.columns]
            all_zero = frame[available_core].fillna(0).eq(0).all(axis=1)
            count_rule(
                rows,
                "all_core_variables_zero",
                "core",
                len(frame),
                int(all_zero.sum()),
                f"{source}; core={','.join(available_core)}",
            )

            work = frame.copy()
            work.loc[all_zero, available_core] = np.nan

            rule_specs = [
                ("o3_le_0", "O3", lambda s: s <= 0),
                ("o3_gt_250", "O3", lambda s: s > 250),
                ("temp_lt_5", "AMB_TEMP", lambda s: s < 5),
                ("temp_gt_45", "AMB_TEMP", lambda s: s > 45),
                ("rh_le_0", "RH", lambda s: s <= 0),
                ("rh_gt_100", "RH", lambda s: s > 100),
                ("wind_speed_lt_0", "WIND_SPEED", lambda s: s < 0),
                ("wind_speed_gt_30", "WIND_SPEED", lambda s: s > 30),
                ("wind_direction_lt_0", "WIND_DIREC", lambda s: s < 0),
                ("wind_direction_gt_360", "WIND_DIREC", lambda s: s > 360),
                ("rainfall_lt_0", "RAINFALL", lambda s: s < 0),
                ("rainfall_gt_500", "RAINFALL", lambda s: s > 500),
            ]
            for rule, col, fn in rule_specs:
                if col not in work.columns:
                    continue
                valid = work[col].notna()
                mask = valid & fn(work[col])
                count_rule(rows, rule, col, int(valid.sum()), int(mask.sum()), source)
                if rule == "wind_speed_gt_30" and mask.any():
                    rec = work.loc[mask, ["site", "datetime", "WIND_SPEED", "WIND_DIREC"]].copy()
                    rec["source_archive"] = source
                    rec["year"] = year
                    wind_gt30_records.append(rec)
                work.loc[mask, col] = np.nan

            for col in ["NO", "NO2", "NOx", "CO", "PM10", "PM2.5"]:
                if col not in work.columns:
                    continue
                valid = work[col].notna()
                mask = valid & (work[col] < 0)
                count_rule(rows, "negative_copollutant", col, int(valid.sum()), int(mask.sum()), source)
                work.loc[mask, col] = np.nan

    audit = pd.DataFrame(rows)
    grouped = (
        audit.groupby(["task", "rule", "variable"], as_index=False)
        .agg(n_checked=("n_checked", "sum"), n_deleted=("n_deleted", "sum"))
    )
    grouped["deleted_pct_of_checked"] = grouped["n_deleted"] / grouped["n_checked"] * 100
    grouped.to_csv(TABLES / "T6_physical_qc_audit_summary.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(TABLES / "T6_physical_qc_audit_by_archive.csv", index=False, encoding="utf-8-sig")

    if wind_gt30_records:
        wind = pd.concat(wind_gt30_records, ignore_index=True)
    else:
        wind = pd.DataFrame(columns=["site", "datetime", "WIND_SPEED", "WIND_DIREC", "source_archive", "year"])
    wind.to_csv(TABLES / "T6_wind_speed_gt30_records.csv", index=False, encoding="utf-8-sig")
    return grouped, wind


def validate_sunshine() -> tuple[pd.DataFrame, pd.DataFrame]:
    sunshine = pd.read_csv(ROOT / "outputs" / "tables" / "kaohsiung_sunshine_daily_2016_2025_extended.csv", parse_dates=["date"])
    sunshine["year"] = sunshine["date"].dt.year
    source_summary = (
        sunshine.groupby(["year", "source_station_id", "source"], as_index=False)
        .agg(n_days=("date", "count"), missing_days=("sunshine_hours", lambda s: int(s.isna().sum())))
        .sort_values(["year", "source_station_id"])
    )
    source_summary.to_csv(TABLES / "T7_sunshine_source_by_year.csv", index=False, encoding="utf-8-sig")

    daily = pd.read_csv(ROOT / "outputs" / "tables" / "daily_mda8_o3_2018_2025.csv", parse_dates=["date"])
    cwa = pd.read_csv(ROOT / "outputs" / "tables" / "cwa_kaohsiung_daily_meteorology_2025.csv", parse_dates=["date"])
    city = daily[daily["date"].dt.year == 2025].groupby("date", as_index=False).agg(merged_sunshine_hours=("sunshine_hours", "mean"))
    merged = city.merge(cwa[["date", "station_id", "cwa_sunshine_hours"]], on="date", how="inner")
    valid = merged[["merged_sunshine_hours", "cwa_sunshine_hours"]].dropna()
    if valid.empty:
        metrics = pd.DataFrame(
            [{"metric": "sunshine_hours_2025", "n_days": 0, "pearson_r": np.nan, "mean_bias_merged_minus_cwa": np.nan, "mae": np.nan}]
        )
    else:
        diff = valid["merged_sunshine_hours"] - valid["cwa_sunshine_hours"]
        metrics = pd.DataFrame(
            [
                {
                    "metric": "sunshine_hours_2025",
                    "source_in_daily_table": "merged sunshine_hours",
                    "reference": "CWA 467441 cwa_sunshine_hours",
                    "n_days": len(valid),
                    "pearson_r": valid["merged_sunshine_hours"].corr(valid["cwa_sunshine_hours"]),
                    "mean_bias_merged_minus_cwa": diff.mean(),
                    "mae": diff.abs().mean(),
                }
            ]
        )
    metrics.to_csv(TABLES / "T7_sunshine_2025_consistency_metrics.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(TABLES / "T7_sunshine_2025_comparison_daily.csv", index=False, encoding="utf-8-sig")
    return source_summary, metrics


def mann_whitney_effect(event_values: pd.Series, non_event_values: pd.Series) -> dict[str, float]:
    x = event_values.dropna().to_numpy(float)
    y = non_event_values.dropna().to_numpy(float)
    if len(x) == 0 or len(y) == 0:
        return {"n_event": len(x), "n_non_event": len(y), "u_stat": np.nan, "p_value": np.nan, "rank_biserial": np.nan}
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    rank_biserial = (2 * res.statistic / (len(x) * len(y))) - 1
    return {
        "n_event": len(x),
        "n_non_event": len(y),
        "u_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "rank_biserial": float(rank_biserial),
    }


def event_redefinition() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(ROOT / "outputs" / "tables" / "daily_mda8_o3_2018_2025.csv", parse_dates=["date"])
    daily["season"] = daily["date"].dt.month.map(month_to_season)
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month

    city = (
        daily.groupby("date", as_index=False)
        .agg(
            city_mean_mda8=("mda8_o3", "mean"),
            city_max_mda8=("mda8_o3", "max"),
            n_sites=("mda8_o3", lambda s: int(s.notna().sum())),
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
    p90 = city["city_max_mda8"].quantile(0.90)
    p95 = city["city_max_mda8"].quantile(0.95)
    city["extreme_o3_event_p90"] = city["city_max_mda8"] >= p90
    city["extreme_o3_event_p95"] = city["city_max_mda8"] >= p95
    city["exceedance_day_60ppb"] = city["city_max_mda8"] >= 60.0
    city.to_csv(TABLES / "T5_city_day_event_redefined.csv", index=False, encoding="utf-8-sig")

    definitions = [
        ("extreme_o3_event_p90", p90, "citywide daily maximum MDA8 O3 >= P90"),
        ("extreme_o3_event_p95", p95, "citywide daily maximum MDA8 O3 >= P95"),
        ("exceedance_day_60ppb", 60.0, "citywide daily maximum MDA8 O3 >= 60 ppb"),
    ]
    prevalence_rows = []
    for label, threshold, definition in definitions:
        for group_name, group_cols in [
            ("overall", []),
            ("year", ["year"]),
            ("season", ["season"]),
            ("year_season", ["year", "season"]),
        ]:
            if group_cols:
                grouped = city.groupby(group_cols)
            else:
                grouped = [((), city)]
            for key, g in grouped:
                rec = {
                    "event_label": label,
                    "definition": definition,
                    "threshold_ppb": threshold,
                    "group": group_name,
                    "n_days": len(g),
                    "n_event_days": int(g[label].sum()),
                    "event_fraction": float(g[label].mean()),
                }
                if group_cols:
                    if not isinstance(key, tuple):
                        key = (key,)
                    for col, val in zip(group_cols, key):
                        rec[col] = val
                prevalence_rows.append(rec)
    prevalence = pd.DataFrame(prevalence_rows)
    prevalence.to_csv(TABLES / "T5_event_prevalence_by_definition.csv", index=False, encoding="utf-8-sig")

    feature_cols = ["temp_max", "rh_mean", "wind_speed_mean", "rain_sum", "sunshine_hours"]
    comparison_rows = []
    city_anom = city.copy()
    for feature in feature_cols:
        city_anom[f"{feature}_month_anom"] = city[feature] - city.groupby("month")[feature].transform("mean")
    for label, threshold, definition in definitions:
        event_mask = city[label]
        for feature in feature_cols:
            stats_raw = mann_whitney_effect(city.loc[event_mask, feature], city.loc[~event_mask, feature])
            comparison_rows.append(
                {
                    "event_label": label,
                    "definition": definition,
                    "threshold_ppb": threshold,
                    "feature": feature,
                    "scale": "raw_city_day",
                    "event_median": city.loc[event_mask, feature].median(),
                    "non_event_median": city.loc[~event_mask, feature].median(),
                    "event_minus_non_event_median": city.loc[event_mask, feature].median() - city.loc[~event_mask, feature].median(),
                    **stats_raw,
                }
            )
            anom = f"{feature}_month_anom"
            stats_anom = mann_whitney_effect(city_anom.loc[event_mask, anom], city_anom.loc[~event_mask, anom])
            comparison_rows.append(
                {
                    "event_label": label,
                    "definition": definition,
                    "threshold_ppb": threshold,
                    "feature": feature,
                    "scale": "month_adjusted_city_day",
                    "event_median": city_anom.loc[event_mask, anom].median(),
                    "non_event_median": city_anom.loc[~event_mask, anom].median(),
                    "event_minus_non_event_median": city_anom.loc[event_mask, anom].median()
                    - city_anom.loc[~event_mask, anom].median(),
                    **stats_anom,
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(TABLES / "T5_event_meteorology_mannwhitney_effects.csv", index=False, encoding="utf-8-sig")

    max_by_date = city[["date", "city_max_mda8", "extreme_o3_event_p90", "exceedance_day_60ppb"]]
    contrib = daily.merge(max_by_date, on="date", how="inner")
    contrib = contrib[np.isclose(contrib["mda8_o3"], contrib["city_max_mda8"], equal_nan=False)].copy()
    contrib["context"] = "all_days"
    contrib_p90 = contrib[contrib["extreme_o3_event_p90"]].copy()
    contrib_p90["context"] = "extreme_o3_event_p90"
    contrib_60 = contrib[contrib["exceedance_day_60ppb"]].copy()
    contrib_60["context"] = "exceedance_day_60ppb"
    all_contrib = pd.concat([contrib, contrib_p90, contrib_60], ignore_index=True)
    station_contrib = (
        all_contrib.groupby(["context", "site"], as_index=False)
        .agg(n_city_max_contributions=("date", "count"), mean_city_max_mda8=("city_max_mda8", "mean"))
        .sort_values(["context", "n_city_max_contributions"], ascending=[True, False])
    )
    station_contrib.to_csv(TABLES / "T5_city_max_contributing_station_counts.csv", index=False, encoding="utf-8-sig")
    return city, prevalence, comparisons, station_contrib


def write_summary(qc: pd.DataFrame, wind: pd.DataFrame, sunshine_source: pd.DataFrame, sunshine_metrics: pd.DataFrame,
                  prevalence: pd.DataFrame, comparisons: pd.DataFrame, station_contrib: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Revision analysis summary (T5/T6/T7)")
    lines.append("")
    lines.append("## T6 資料清理稽核")
    total_deleted = int(qc["n_deleted"].sum())
    lines.append(f"- 已依原始年度壓縮檔重建 QC 稽核表，逐規則合計被設為缺值或刪除的值共 {total_deleted:,} 筆次。")
    top = qc.sort_values("n_deleted", ascending=False).head(8)
    lines.append("")
    lines.append(top.to_markdown(index=False))
    lines.append("")
    lines.append(f"- 風速 >30 m/s 的紀錄數：{len(wind):,}。詳細日期已輸出至 `T6_wind_speed_gt30_records.csv`。")
    lines.append("")

    lines.append("## T7 日照資料一致性")
    lines.append("- 日照來源按年份/測站統計已輸出至 `T7_sunshine_source_by_year.csv`。")
    lines.append(sunshine_source.to_markdown(index=False))
    lines.append("")
    lines.append("- 2025 merged sunshine 與 CWA 467441 日照比對：")
    lines.append(sunshine_metrics.to_markdown(index=False))
    lines.append("")

    lines.append("## T5 事件定義重構")
    overall = prevalence[prevalence["group"] == "overall"].copy()
    lines.append(overall[["event_label", "threshold_ppb", "n_days", "n_event_days", "event_fraction"]].to_markdown(index=False))
    lines.append("")
    p90 = comparisons[(comparisons["event_label"] == "extreme_o3_event_p90") & (comparisons["scale"] == "month_adjusted_city_day")]
    lines.append("- P90 extreme O3 event 的月調整氣象差異與 Mann-Whitney U / rank-biserial effect size：")
    lines.append(
        p90[[
            "feature",
            "event_median",
            "non_event_median",
            "event_minus_non_event_median",
            "p_value",
            "rank_biserial",
        ]].to_markdown(index=False)
    )
    lines.append("")
    lines.append("- 全市日最大 MDA8 O3 的主要貢獻測站前幾名：")
    lines.append(station_contrib.groupby("context").head(5).to_markdown(index=False))
    lines.append("")
    (OUT / "results_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("[revision] Running T6 physical QC audit", flush=True)
    qc, wind = audit_physical_qc()
    print("[revision] Running T7 sunshine validation", flush=True)
    sunshine_source, sunshine_metrics = validate_sunshine()
    print("[revision] Running T5 event redefinition", flush=True)
    _, prevalence, comparisons, station_contrib = event_redefinition()
    write_summary(qc, wind, sunshine_source, sunshine_metrics, prevalence, comparisons, station_contrib)
    print(f"[revision] Wrote outputs to {OUT}", flush=True)


if __name__ == "__main__":
    main()
