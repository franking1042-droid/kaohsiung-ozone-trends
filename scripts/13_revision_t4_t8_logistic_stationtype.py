from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


DEFAULT_OUT = Path("analysis_outputs/revision/T4_T8_logistic_stationtype")

LOGISTIC_FEATURES = [
    "temp_max",
    "rh_mean",
    "wind_speed_mean",
    "rain_log1p",
    "sunshine_hours",
    "day_wind_u",
    "day_wind_v",
]

EVENT_LABELS = [
    "extreme_o3_event_p90",
    "extreme_o3_event_p95",
    "exceedance_day_60ppb",
]

SITE_TYPE_EN = {
    "一般站": "general",
    "交通站": "traffic",
    "工業站": "industrial",
    "背景站": "background",
    "其它站": "other",
}

TYPE_ORDER = ["background", "general", "industrial", "traffic", "other"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T4 logistic and T8 station-type contrast analyses.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--city-events", default="analysis_outputs/revision/tables/T5_city_day_event_redefined.csv")
    parser.add_argument("--city-model", default="analysis_outputs/revision/T2_T3_models/tables/T2_city_day_model_dataset.csv")
    parser.add_argument("--station-day", default="analysis_outputs/revision/T2_T3_models/tables/T3_daily_mda8_o3_with_daynight_wind.csv")
    parser.add_argument("--station-meta", default="outputs/tables/kaohsiung_air_quality_station_metadata.csv")
    parser.add_argument("--t1-station-trends", default="analysis_outputs/revision/T1_met_norm_main300/trend_results_by_station.csv")
    parser.add_argument("--city-max-contrib", default="analysis_outputs/revision/tables/T5_city_max_contributing_station_counts.csv")
    return parser.parse_args()


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def load_t4_city_data(city_events_path: Path, city_model_path: Path) -> pd.DataFrame:
    events = read_csv(city_events_path, parse_dates=["date"])
    city = read_csv(city_model_path, parse_dates=["date"])
    keep = ["date", "city_mean_mda8", "city_max_mda8", "n_sites"] + EVENT_LABELS
    df = city.merge(events[keep], on="date", how="inner", suffixes=("", "_event"))
    for col in EVENT_LABELS:
        df[col] = df[col].astype(bool)
    if "rain_log1p" not in df.columns:
        df["rain_log1p"] = np.log1p(df["rain_sum"].clip(lower=0))
    return ensure_numeric(df, LOGISTIC_FEATURES + ["year", "month"])


def standardize_frame(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    x = df[features].astype(float).copy()
    stats: dict[str, float] = {}
    for col in features:
        mean = float(x[col].mean())
        sd = float(x[col].std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        x[col] = (x[col] - mean) / sd
        stats[f"{col}_mean"] = mean
        stats[f"{col}_sd"] = sd
    return x, stats


def make_scaled_city_data(df: pd.DataFrame, scale: str) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    if scale == "raw_city_day":
        return data, LOGISTIC_FEATURES.copy()
    if scale != "month_adjusted_city_day":
        raise ValueError(f"Unknown scale: {scale}")
    adjusted_features = []
    for feature in LOGISTIC_FEATURES:
        adj = f"{feature}_month_anomaly"
        data[adj] = data[feature] - data.groupby("month")[feature].transform("mean")
        adjusted_features.append(adj)
    return data, adjusted_features


def fit_one_logit(data: pd.DataFrame, event_label: str, scale: str) -> tuple[pd.DataFrame, dict[str, object]]:
    scaled_data, features = make_scaled_city_data(data, scale)
    model_data = scaled_data.dropna(subset=[event_label] + features).copy()
    y = model_data[event_label].astype(int)
    x_std, stats = standardize_frame(model_data, features)
    x = sm.add_constant(x_std, has_constant="add")

    note = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sm.Logit(y, x).fit(disp=False, maxiter=500)
        if caught:
            note = "; ".join(sorted({str(w.message) for w in caught}))

    params = fit.params.drop("const", errors="ignore")
    conf = fit.conf_int().drop("const", errors="ignore")
    rows = []
    for feature in params.index:
        clean_feature = feature.replace("_month_anomaly", "")
        rows.append(
            {
                "event_label": event_label,
                "scale": scale,
                "feature": clean_feature,
                "model_feature": feature,
                "coefficient_per_sd": float(params[feature]),
                "std_error": float(fit.bse[feature]),
                "z": float(fit.tvalues[feature]),
                "p_value": float(fit.pvalues[feature]),
                "odds_ratio_per_sd": float(np.exp(params[feature])),
                "or_ci_low": float(np.exp(conf.loc[feature, 0])),
                "or_ci_high": float(np.exp(conf.loc[feature, 1])),
                "feature_mean_for_standardization": stats.get(f"{feature}_mean", np.nan),
                "feature_sd_for_standardization": stats.get(f"{feature}_sd", np.nan),
            }
        )

    pred = fit.predict(x)
    pred_label = (pred >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_label, labels=[0, 1]).ravel()
    null_llf = fit.llnull if np.isfinite(fit.llnull) else np.nan
    pseudo_r2 = 1 - fit.llf / null_llf if np.isfinite(null_llf) and null_llf != 0 else np.nan
    summary = {
        "event_label": event_label,
        "scale": scale,
        "n_days": int(len(model_data)),
        "n_event_days": int(y.sum()),
        "event_fraction": float(y.mean()),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "log_likelihood": float(fit.llf),
        "mcfadden_pseudo_r2": float(pseudo_r2),
        "roc_auc_in_sample": float(roc_auc_score(y, pred)) if y.nunique() == 2 else np.nan,
        "average_precision_in_sample": float(average_precision_score(y, pred)) if y.nunique() == 2 else np.nan,
        "balanced_accuracy_at_0p5": float(balanced_accuracy_score(y, pred_label)),
        "brier_score": float(brier_score_loss(y, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "converged": bool(fit.mle_retvals.get("converged", False)),
        "warning_note": note,
        "features": ",".join(features),
    }
    return pd.DataFrame(rows), summary


def run_t4(city: pd.DataFrame, out_tables: Path, out_figures: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    or_frames = []
    summaries = []
    for event_label in EVENT_LABELS:
        for scale in ["raw_city_day", "month_adjusted_city_day"]:
            try:
                ors, summary = fit_one_logit(city, event_label, scale)
                or_frames.append(ors)
                summaries.append(summary)
                print(f"[T4] fitted {event_label} / {scale}", flush=True)
            except Exception as exc:
                summaries.append(
                    {
                        "event_label": event_label,
                        "scale": scale,
                        "n_days": 0,
                        "n_event_days": 0,
                        "event_fraction": np.nan,
                        "aic": np.nan,
                        "bic": np.nan,
                        "log_likelihood": np.nan,
                        "mcfadden_pseudo_r2": np.nan,
                        "roc_auc_in_sample": np.nan,
                        "average_precision_in_sample": np.nan,
                        "balanced_accuracy_at_0p5": np.nan,
                        "brier_score": np.nan,
                        "tn": np.nan,
                        "fp": np.nan,
                        "fn": np.nan,
                        "tp": np.nan,
                        "converged": False,
                        "warning_note": str(exc),
                        "features": ",".join(LOGISTIC_FEATURES),
                    }
                )
                print(f"[T4] failed {event_label} / {scale}: {exc}", flush=True)

    odds = pd.concat(or_frames, ignore_index=True) if or_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    odds.to_csv(out_tables / "T4_logistic_odds_ratios.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_tables / "T4_logistic_model_summary.csv", index=False, encoding="utf-8-sig")
    make_t4_figures(odds, out_figures)
    return odds, summary_df


def make_t4_figures(odds: pd.DataFrame, out_figures: Path) -> None:
    if odds.empty:
        return
    sns.set_theme(style="whitegrid", context="paper")
    plot = odds[
        (odds["scale"] == "month_adjusted_city_day")
        & (odds["event_label"].isin(["extreme_o3_event_p90", "exceedance_day_60ppb"]))
    ].copy()
    plot["label"] = plot["feature"].replace(
        {
            "temp_max": "Tmax",
            "rh_mean": "RH",
            "wind_speed_mean": "Wind speed",
            "rain_log1p": "Rain log1p",
            "sunshine_hours": "Sunshine",
            "day_wind_u": "Day wind u",
            "day_wind_v": "Day wind v",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharex=False)
    for ax, event_label in zip(axes, ["extreme_o3_event_p90", "exceedance_day_60ppb"]):
        sub = plot[plot["event_label"] == event_label].sort_values("odds_ratio_per_sd")
        y = np.arange(len(sub))
        ax.errorbar(
            sub["odds_ratio_per_sd"],
            y,
            xerr=[
                sub["odds_ratio_per_sd"] - sub["or_ci_low"],
                sub["or_ci_high"] - sub["odds_ratio_per_sd"],
            ],
            fmt="o",
            color="#2b6f8a",
            ecolor="#8ab0bf",
            capsize=3,
        )
        ax.axvline(1.0, color="#333333", linewidth=0.9, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"])
        ax.set_xscale("log")
        ax.set_xlabel("Odds ratio per 1 SD increase")
        ax.set_title(event_label)
    fig.tight_layout()
    fig.savefig(out_figures / "T4_logistic_month_adjusted_odds_ratios.png", dpi=300)
    plt.close(fig)


def load_station_metadata(path: Path) -> pd.DataFrame:
    meta = read_csv(path)
    meta["site_type_en"] = meta["site_type"].map(SITE_TYPE_EN).fillna("other")
    meta["site_label"] = meta["site_eng_name"].fillna(meta["site"])
    return meta


def load_station_day(path: Path, meta: pd.DataFrame, city_events: pd.DataFrame) -> pd.DataFrame:
    df = read_csv(path, parse_dates=["date"])
    numeric = [
        "mda8_o3",
        "temp_max",
        "rh_mean",
        "wind_speed_mean",
        "rain_sum",
        "rain_log1p",
        "sunshine_hours",
        "no2_mean",
        "nox_mean",
        "co_mean",
        "pm25_mean",
        "day_wind_u",
        "day_wind_v",
        "year",
        "month",
    ]
    df = ensure_numeric(df, numeric)
    df = df.merge(
        meta[
            [
                "site",
                "site_eng_name",
                "site_type",
                "site_type_en",
                "longitude",
                "latitude",
                "township",
            ]
        ],
        on="site",
        how="left",
    )
    df = df.merge(city_events[["date"] + EVENT_LABELS], on="date", how="left")
    for col in EVENT_LABELS:
        df[col] = df[col].fillna(False).astype(bool)
    return df


def run_t8(
    station_day: pd.DataFrame,
    meta: pd.DataFrame,
    trends_path: Path,
    contrib_path: Path,
    out_tables: Path,
    out_figures: Path,
) -> dict[str, pd.DataFrame]:
    meta.to_csv(out_tables / "T8_station_metadata_classification.csv", index=False, encoding="utf-8-sig")

    station_summary = station_day.groupby(
        ["site", "site_eng_name", "site_type", "site_type_en", "longitude", "latitude"], as_index=False
    ).agg(
        n_station_days=("mda8_o3", "count"),
        mean_mda8=("mda8_o3", "mean"),
        median_mda8=("mda8_o3", "median"),
        p90_mda8=("mda8_o3", lambda x: x.quantile(0.90)),
        exceed60_fraction=("mda8_o3", lambda x: (x >= 60).mean()),
        mean_no2=("no2_mean", "mean"),
        mean_nox=("nox_mean", "mean"),
        mean_co=("co_mean", "mean"),
        mean_pm25=("pm25_mean", "mean"),
        mean_day_wind_u=("day_wind_u", "mean"),
        mean_day_wind_v=("day_wind_v", "mean"),
        city_p90_event_mean_mda8=("mda8_o3", lambda x: np.nan),
    )
    event_mean = (
        station_day[station_day["extreme_o3_event_p90"]]
        .groupby("site", as_index=False)
        .agg(city_p90_event_mean_mda8=("mda8_o3", "mean"))
    )
    station_summary = station_summary.drop(columns=["city_p90_event_mean_mda8"]).merge(event_mean, on="site", how="left")

    type_summary = station_day.groupby(["site_type", "site_type_en"], as_index=False).agg(
        n_sites=("site", "nunique"),
        n_station_days=("mda8_o3", "count"),
        mean_mda8=("mda8_o3", "mean"),
        median_mda8=("mda8_o3", "median"),
        p90_mda8=("mda8_o3", lambda x: x.quantile(0.90)),
        exceed60_fraction=("mda8_o3", lambda x: (x >= 60).mean()),
        city_p90_event_station_day_fraction=("extreme_o3_event_p90", "mean"),
        mean_no2=("no2_mean", "mean"),
        mean_nox=("nox_mean", "mean"),
        mean_co=("co_mean", "mean"),
        mean_pm25=("pm25_mean", "mean"),
        mean_day_wind_u=("day_wind_u", "mean"),
        mean_day_wind_v=("day_wind_v", "mean"),
    )

    event_rows = []
    for event_label in EVENT_LABELS:
        for (site_type, site_type_en, is_event), group in station_day.groupby(["site_type", "site_type_en", event_label]):
            event_rows.append(
                {
                    "event_label": event_label,
                    "site_type": site_type,
                    "site_type_en": site_type_en,
                    "event_period": "event_day" if bool(is_event) else "non_event_day",
                    "n_station_days": int(group["mda8_o3"].count()),
                    "mean_mda8": group["mda8_o3"].mean(),
                    "median_mda8": group["mda8_o3"].median(),
                    "mean_no2": group["no2_mean"].mean(),
                    "mean_nox": group["nox_mean"].mean(),
                    "mean_co": group["co_mean"].mean(),
                    "mean_pm25": group["pm25_mean"].mean(),
                    "mean_day_wind_u": group["day_wind_u"].mean(),
                    "mean_day_wind_v": group["day_wind_v"].mean(),
                }
            )
    event_type = pd.DataFrame(event_rows)

    contrib = read_csv(contrib_path)
    contrib = contrib.merge(meta[["site", "site_type", "site_type_en", "site_eng_name"]], on="site", how="left")
    contrib_type = (
        contrib.groupby(["context", "site_type", "site_type_en"], as_index=False)
        .agg(n_city_max_contributions=("n_city_max_contributions", "sum"))
        .sort_values(["context", "n_city_max_contributions"], ascending=[True, False])
    )
    contrib_type["context_total"] = contrib_type.groupby("context")["n_city_max_contributions"].transform("sum")
    contrib_type["contribution_fraction"] = contrib_type["n_city_max_contributions"] / contrib_type["context_total"]

    trends = read_csv(trends_path)
    trends = trends[trends["series"] == "met_normalized"].copy()
    trends = ensure_numeric(trends, ["sen_slope_ppb_per_yr", "ci_low", "ci_high", "p_value"])
    trends = trends.merge(meta[["site", "site_type", "site_type_en", "site_eng_name"]], on="site", how="left")
    trend_type = trends.groupby(["site_type", "site_type_en"], as_index=False).agg(
        n_sites=("site", "count"),
        mean_sen_slope_ppb_per_yr=("sen_slope_ppb_per_yr", "mean"),
        median_sen_slope_ppb_per_yr=("sen_slope_ppb_per_yr", "median"),
        n_significant_decreasing=("p_value", lambda x: int((x < 0.05).sum())),
    )

    station_summary.to_csv(out_tables / "T8_station_summary.csv", index=False, encoding="utf-8-sig")
    type_summary.to_csv(out_tables / "T8_station_type_summary.csv", index=False, encoding="utf-8-sig")
    event_type.to_csv(out_tables / "T8_event_vs_nonevent_by_station_type.csv", index=False, encoding="utf-8-sig")
    contrib.to_csv(out_tables / "T8_city_max_contribution_by_station.csv", index=False, encoding="utf-8-sig")
    contrib_type.to_csv(out_tables / "T8_city_max_contribution_by_station_type.csv", index=False, encoding="utf-8-sig")
    trends.to_csv(out_tables / "T8_met_normalized_trend_by_station_type.csv", index=False, encoding="utf-8-sig")
    trend_type.to_csv(out_tables / "T8_met_normalized_trend_summary_by_station_type.csv", index=False, encoding="utf-8-sig")
    make_t8_figures(station_day, station_summary, type_summary, contrib_type, trends, out_figures)
    print("[T8] station-type outputs written", flush=True)
    return {
        "station_summary": station_summary,
        "type_summary": type_summary,
        "event_type": event_type,
        "contrib_type": contrib_type,
        "trends": trends,
        "trend_type": trend_type,
    }


def ordered_type_series(series: pd.Series) -> pd.Categorical:
    observed = [x for x in TYPE_ORDER if x in set(series.dropna())]
    extras = sorted(set(series.dropna()) - set(observed))
    return pd.Categorical(series, categories=observed + extras, ordered=True)


def make_t8_figures(
    station_day: pd.DataFrame,
    station_summary: pd.DataFrame,
    type_summary: pd.DataFrame,
    contrib_type: pd.DataFrame,
    trends: pd.DataFrame,
    out_figures: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plot_day = station_day.dropna(subset=["mda8_o3", "site_type_en"]).copy()
    plot_day["site_type_en"] = ordered_type_series(plot_day["site_type_en"])

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    sns.boxplot(data=plot_day, x="site_type_en", y="mda8_o3", ax=ax, showfliers=False, color="#9fc1b5")
    ax.set_xlabel("Station type")
    ax.set_ylabel("Station-day MDA8 O3 (ppb)")
    ax.set_title("MDA8 O3 distribution by station type")
    fig.tight_layout()
    fig.savefig(out_figures / "T8_station_type_mda8_boxplot.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    scatter = station_summary.dropna(subset=["mean_nox", "mean_mda8"]).copy()
    sns.scatterplot(
        data=scatter,
        x="mean_nox",
        y="mean_mda8",
        hue="site_type_en",
        style="site_type_en",
        s=90,
        ax=ax,
    )
    for _, row in scatter.iterrows():
        ax.text(row["mean_nox"], row["mean_mda8"], str(row["site_eng_name"]), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Mean NOx (ppb)")
    ax.set_ylabel("Mean MDA8 O3 (ppb)")
    ax.set_title("Station contrast: NOx and MDA8 O3")
    fig.tight_layout()
    fig.savefig(out_figures / "T8_station_mean_nox_vs_o3.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    cplot = contrib_type[contrib_type["context"].isin(["all_days", "extreme_o3_event_p90", "exceedance_day_60ppb"])].copy()
    sns.barplot(data=cplot, x="context", y="contribution_fraction", hue="site_type_en", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of city-maximum contributions")
    ax.set_title("City daily maximum MDA8 O3 contribution by station type")
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(out_figures / "T8_city_max_contribution_by_station_type.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    tplot = trends.sort_values("sen_slope_ppb_per_yr").copy()
    sns.barplot(data=tplot, x="site_eng_name", y="sen_slope_ppb_per_yr", hue="site_type_en", dodge=False, ax=ax)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Station")
    ax.set_ylabel("Met-normalized Sen slope (ppb/year)")
    ax.set_title("Meteorologically normalized trends by station type")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(out_figures / "T8_station_type_met_normalized_trends.png", dpi=300)
    plt.close(fig)


def write_markdown_summary(
    out_dir: Path,
    odds: pd.DataFrame,
    logit_summary: pd.DataFrame,
    t8: dict[str, pd.DataFrame],
) -> None:
    p90_month = odds[
        (odds["event_label"] == "extreme_o3_event_p90")
        & (odds["scale"] == "month_adjusted_city_day")
    ].sort_values("p_value")
    exceed_month = odds[
        (odds["event_label"] == "exceedance_day_60ppb")
        & (odds["scale"] == "month_adjusted_city_day")
    ].sort_values("p_value")
    contrib_p90 = t8["contrib_type"][t8["contrib_type"]["context"] == "extreme_o3_event_p90"].copy()
    trend_type = t8["trend_type"].copy()
    type_summary = t8["type_summary"].copy()

    lines = [
        "# T4/T8 analysis summary",
        "",
        "## T4 unified logistic regression",
        "",
        "- Logistic odds ratios were fitted with a single statsmodels Logit MLE model.",
        "- ORs are reported per 1 SD increase in each predictor.",
        "- Both raw city-day predictors and month-adjusted city-day anomalies were fitted.",
        "",
        "Top month-adjusted predictors for P90 extreme O3 events:",
    ]
    for _, row in p90_month.head(5).iterrows():
        lines.append(
            f"- {row['feature']}: OR={row['odds_ratio_per_sd']:.2f} "
            f"({row['or_ci_low']:.2f}-{row['or_ci_high']:.2f}), p={row['p_value']:.3g}"
        )
    lines += ["", "Top month-adjusted predictors for 60 ppb exceedance days:"]
    for _, row in exceed_month.head(5).iterrows():
        lines.append(
            f"- {row['feature']}: OR={row['odds_ratio_per_sd']:.2f} "
            f"({row['or_ci_low']:.2f}-{row['or_ci_high']:.2f}), p={row['p_value']:.3g}"
        )

    lines += [
        "",
        "## T8 station-type contrast",
        "",
        "Station-type descriptive summary:",
    ]
    for _, row in type_summary.sort_values("mean_mda8", ascending=False).iterrows():
        lines.append(
            f"- {row['site_type_en']} ({row['site_type']}): n_sites={int(row['n_sites'])}, "
            f"mean MDA8={row['mean_mda8']:.1f} ppb, exceed60 fraction={row['exceed60_fraction']:.2f}, "
            f"mean NOx={row['mean_nox']:.1f} ppb"
        )

    lines += ["", "P90 city-maximum contribution by station type:"]
    for _, row in contrib_p90.sort_values("n_city_max_contributions", ascending=False).iterrows():
        lines.append(
            f"- {row['site_type_en']} ({row['site_type']}): "
            f"{int(row['n_city_max_contributions'])} days "
            f"({row['contribution_fraction']:.1%})"
        )

    lines += ["", "Met-normalized trend by station type:"]
    for _, row in trend_type.sort_values("mean_sen_slope_ppb_per_yr").iterrows():
        lines.append(
            f"- {row['site_type_en']} ({row['site_type']}): "
            f"mean slope={row['mean_sen_slope_ppb_per_yr']:.3f} ppb/year, "
            f"significant decreasing sites={int(row['n_significant_decreasing'])}/{int(row['n_sites'])}"
        )

    lines += [
        "",
        "## Outputs",
        "",
        "- Tables: `analysis_outputs/revision/T4_T8_logistic_stationtype/tables/`",
        "- Figures: `analysis_outputs/revision/T4_T8_logistic_stationtype/figures/`",
    ]
    (out_dir / "T4_T8分析摘要.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_tables = out_dir / "tables"
    out_figures = out_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    city_events = read_csv(args.city_events, parse_dates=["date"])
    city = load_t4_city_data(Path(args.city_events), Path(args.city_model))
    odds, logit_summary = run_t4(city, out_tables, out_figures)

    meta = load_station_metadata(Path(args.station_meta))
    station_day = load_station_day(Path(args.station_day), meta, city_events)
    t8 = run_t8(
        station_day=station_day,
        meta=meta,
        trends_path=Path(args.t1_station_trends),
        contrib_path=Path(args.city_max_contrib),
        out_tables=out_tables,
        out_figures=out_figures,
    )
    write_markdown_summary(out_dir, odds, logit_summary, t8)
    print(f"[T4/T8] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
