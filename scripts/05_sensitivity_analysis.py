from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pygam import LinearGAM, f, s
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
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
EVENT_FEATURES = ["mda8_o3", "temp_max", "rh_mean", "wind_speed_mean", "rain_sum", "sunshine_hours"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sensitivity analyses for Kaohsiung MDA8 O3 study.")
    parser.add_argument("--daily", default="outputs/tables/daily_mda8_o3_2018_2025.csv")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--rf-trees", type=int, default=300)
    return parser.parse_args()


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
    df["rain_log1p"] = np.log1p(df["rain_sum"].clip(lower=0))
    df["year_index"] = df["year"] - df["year"].min()
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(month_to_season)
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(df["year"].unique())
    if len(years) >= 4:
        split_year = years[-2]
        train = df[df["year"] < split_year].copy()
        test = df[df["year"] >= split_year].copy()
    else:
        train = df.sample(frac=0.8, random_state=42)
        test = df.drop(train.index)
    return train, test


def model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def fit_rf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    n_estimators: int,
    experiment: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    rf_features = features + ["site"]
    X_train = pd.get_dummies(train[rf_features], columns=["site"], drop_first=False).astype(float)
    X_test = pd.get_dummies(test[rf_features], columns=["site"], drop_first=False)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0).astype(float)
    y_train = train["mda8_o3"].to_numpy()
    y_test = test["mda8_o3"].to_numpy()

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    result = model_metrics(y_test, pred)
    result.update(
        {
            "experiment": experiment,
            "model": "Random Forest",
            "n_train": len(train),
            "n_test": len(test),
            "train_years": f"{train['year'].min()}-{train['year'].max()}",
            "test_years": f"{test['year'].min()}-{test['year'].max()}",
            "oob_r2": getattr(model, "oob_score_", np.nan),
        }
    )

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=42,
        n_jobs=-1,
    )
    imp = pd.DataFrame(
        {
            "experiment": experiment,
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_sd": importance.importances_std,
        }
    ).sort_values(["experiment", "importance_mean"], ascending=[True, False])
    return result, imp


def fit_gam(train: pd.DataFrame, test: pd.DataFrame, features: list[str], experiment: str) -> dict[str, object]:
    gam_features = features + ["site_code"]
    train = train.copy()
    test = test.copy()
    categories = pd.Categorical(train["site"])
    site_map = {site: code for code, site in enumerate(categories.categories)}
    train["site_code"] = train["site"].map(site_map).astype(float)
    test["site_code"] = test["site"].map(site_map).fillna(-1).astype(float)

    X_train = train[gam_features].to_numpy()
    X_test = test[gam_features].to_numpy()
    y_train = train["mda8_o3"].to_numpy()
    y_test = test["mda8_o3"].to_numpy()

    terms = s(0, n_splines=10)
    for idx in range(1, len(features)):
        terms += s(idx, n_splines=10)
    terms += f(len(features))

    gam = LinearGAM(terms).fit(X_train, y_train)
    pred = gam.predict(X_test)
    result = model_metrics(y_test, pred)
    result.update(
        {
            "experiment": experiment,
            "model": "GAM",
            "n_train": len(train),
            "n_test": len(test),
            "train_years": f"{train['year'].min()}-{train['year'].max()}",
            "test_years": f"{test['year'].min()}-{test['year'].max()}",
            "oob_r2": np.nan,
        }
    )
    return result


def run_model_sensitivity(df: pd.DataFrame, tables_dir: Path, figures_dir: Path, rf_trees: int) -> None:
    configs = [
        {
            "experiment": "base_full_2018_2025",
            "features": BASE_FEATURES,
            "seasons": None,
            "require_sunshine": False,
        },
        {
            "experiment": "precursors_full_2018_2025",
            "features": BASE_FEATURES + PRECURSOR_FEATURES,
            "seasons": None,
            "require_sunshine": False,
        },
        {
            "experiment": "base_spring_autumn_2018_2025",
            "features": BASE_FEATURES,
            "seasons": ["spring", "autumn"],
            "require_sunshine": False,
        },
        {
            "experiment": "precursors_spring_autumn_2018_2025",
            "features": BASE_FEATURES + PRECURSOR_FEATURES,
            "seasons": ["spring", "autumn"],
            "require_sunshine": False,
        },
        {
            "experiment": "sunshine_full_2018_2025",
            "features": BASE_FEATURES + ["sunshine_hours"],
            "seasons": None,
            "require_sunshine": True,
        },
    ]

    performance_rows: list[dict[str, object]] = []
    importance_frames: list[pd.DataFrame] = []

    for config in configs:
        experiment = config["experiment"]
        features = list(config["features"])
        subset = df.copy()
        if config["seasons"] is not None:
            subset = subset[subset["season"].isin(config["seasons"])].copy()
        if config["require_sunshine"]:
            subset = subset[subset["sunshine_hours"].notna()].copy()

        subset = subset.dropna(subset=["mda8_o3", "site"] + features).copy()
        if subset.empty:
            continue

        train, test = temporal_split(subset)
        print(
            f"{experiment}: rows={len(subset)}, train={train['year'].min()}-{train['year'].max()}, "
            f"test={test['year'].min()}-{test['year'].max()}",
            flush=True,
        )

        rf_result, rf_importance = fit_rf(train, test, features, rf_trees, experiment)
        gam_result = fit_gam(train, test, features, experiment)
        for row in (rf_result, gam_result):
            row["features"] = ",".join(features)
            row["seasons"] = "all" if config["seasons"] is None else ",".join(config["seasons"])
        performance_rows.extend([rf_result, gam_result])
        importance_frames.append(rf_importance)

    performance = pd.DataFrame(performance_rows)
    performance = performance[
        [
            "experiment",
            "model",
            "seasons",
            "features",
            "n_train",
            "n_test",
            "train_years",
            "test_years",
            "r2",
            "rmse",
            "mae",
            "oob_r2",
        ]
    ]
    performance.to_csv(tables_dir / "sensitivity_model_performance.csv", index=False, encoding="utf-8-sig")
    pd.concat(importance_frames, ignore_index=True).to_csv(
        tables_dir / "sensitivity_rf_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    plot_df = performance.copy()
    plot_df["experiment"] = plot_df["experiment"].str.replace("_", "\n")
    sns.barplot(data=plot_df, x="experiment", y="r2", hue="model", ax=ax)
    ax.set_title("Sensitivity model performance")
    ax.set_xlabel("")
    ax.set_ylabel("Test R2")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "sensitivity_model_r2.png", dpi=220)
    plt.close(fig)


def summarize_event_quantile(df: pd.DataFrame, quantile: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    city = (
        df.groupby("date", as_index=False)
        .agg(
            city_mean_mda8=("mda8_o3", "mean"),
            city_max_mda8=("mda8_o3", "max"),
            affected_sites=("mda8_o3", lambda x: x.notna().sum()),
        )
        .dropna(subset=["city_max_mda8"])
    )
    threshold = city["city_max_mda8"].quantile(quantile)
    event_dates = set(city.loc[city["city_max_mda8"] >= threshold, "date"])

    event_days = city[city["date"].isin(event_dates)].copy()
    event_days["quantile"] = quantile
    event_days["threshold_ppb"] = threshold
    event_days["year"] = event_days["date"].dt.year
    event_days["month"] = event_days["date"].dt.month
    event_days["season"] = event_days["month"].map(month_to_season)

    counts = pd.DataFrame(
        [
            {
                "quantile": quantile,
                "threshold_ppb": threshold,
                "n_event_days": len(event_days),
                "city_max_mda8_mean": event_days["city_max_mda8"].mean(),
                "city_max_mda8_max": event_days["city_max_mda8"].max(),
            }
        ]
    )

    daily = df.copy()
    daily["high_o3_event_day"] = daily["date"].isin(event_dates)
    summary_rows = []
    anomaly_rows = []
    available_features = [feature for feature in EVENT_FEATURES if feature in daily.columns]
    for feature in available_features:
        for is_event, values in daily.groupby("high_o3_event_day")[feature]:
            summary_rows.append(
                {
                    "quantile": quantile,
                    "threshold_ppb": threshold,
                    "feature": feature,
                    "period": "event" if is_event else "non_event",
                    "mean": values.mean(),
                    "median": values.median(),
                    "p25": values.quantile(0.25),
                    "p75": values.quantile(0.75),
                    "n": values.count(),
                }
            )

        climatology = daily.groupby(["site", "month"])[feature].transform("mean")
        anomaly = daily[feature] - climatology
        tmp = pd.DataFrame({"event": daily["high_o3_event_day"], "anomaly": anomaly})
        for is_event, values in tmp.groupby("event")["anomaly"]:
            anomaly_rows.append(
                {
                    "quantile": quantile,
                    "threshold_ppb": threshold,
                    "feature": feature,
                    "period": "event" if is_event else "non_event",
                    "monthly_adjusted_mean": values.mean(),
                    "monthly_adjusted_median": values.median(),
                    "monthly_adjusted_p25": values.quantile(0.25),
                    "monthly_adjusted_p75": values.quantile(0.75),
                    "n": values.count(),
                }
            )

    return counts, event_days, pd.DataFrame(summary_rows), pd.DataFrame(anomaly_rows)


def run_event_sensitivity(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    counts_frames = []
    event_day_frames = []
    summary_frames = []
    anomaly_frames = []
    for quantile in (0.90, 0.95):
        counts, event_days, summary, anomaly = summarize_event_quantile(df, quantile)
        counts_frames.append(counts)
        event_day_frames.append(event_days)
        summary_frames.append(summary)
        anomaly_frames.append(anomaly)

    counts = pd.concat(counts_frames, ignore_index=True)
    event_days = pd.concat(event_day_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    anomaly = pd.concat(anomaly_frames, ignore_index=True)

    counts.to_csv(tables_dir / "sensitivity_event_thresholds.csv", index=False, encoding="utf-8-sig")
    event_days.to_csv(tables_dir / "sensitivity_event_days.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(tables_dir / "sensitivity_event_meteorology_summary.csv", index=False, encoding="utf-8-sig")
    anomaly.to_csv(tables_dir / "sensitivity_event_meteorology_month_adjusted.csv", index=False, encoding="utf-8-sig")

    by_month = (
        event_days.groupby(["quantile", "threshold_ppb", "month"], as_index=False)
        .size()
        .rename(columns={"size": "n_event_days"})
    )
    by_season = (
        event_days.groupby(["quantile", "threshold_ppb", "season"], as_index=False)
        .size()
        .rename(columns={"size": "n_event_days"})
    )
    by_month.to_csv(tables_dir / "sensitivity_event_counts_by_month.csv", index=False, encoding="utf-8-sig")
    by_season.to_csv(tables_dir / "sensitivity_event_counts_by_season.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.barplot(data=by_season, x="season", y="n_event_days", hue="quantile", ax=ax)
    ax.set_title("High O3 event counts by threshold and season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Event days")
    fig.tight_layout()
    fig.savefig(figures_dir / "sensitivity_event_counts_by_season.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    df = load_daily(Path(args.daily))
    run_event_sensitivity(df, tables_dir, figures_dir)
    run_model_sensitivity(df, tables_dir, figures_dir, args.rf_trees)
    print("Wrote sensitivity outputs.", flush=True)


if __name__ == "__main__":
    main()
