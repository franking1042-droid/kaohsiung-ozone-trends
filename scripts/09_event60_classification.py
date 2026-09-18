from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None


BASE_FEATURES = [
    "temp_max",
    "rh_mean",
    "wind_speed_mean",
    "rain_log1p",
    "wind_dir_sin",
    "wind_dir_cos",
    "doy_sin",
    "doy_cos",
    "year_index",
]
SUNSHINE_FEATURES = BASE_FEATURES + ["sunshine_hours"]
PRECURSOR_FEATURES = SUNSHINE_FEATURES + ["nox_mean", "no2_mean", "co_mean", "pm25_mean"]
EVENT_SUMMARY_FEATURES = ["temp_max", "rh_mean", "wind_speed_mean", "rain_sum", "sunshine_hours"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-threshold O3 event definition and classification models.")
    parser.add_argument("--daily", default="outputs/tables/daily_mda8_o3_2018_2025.csv")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--threshold", type=float, default=60.0, help="MDA8 O3 event threshold in ppb.")
    parser.add_argument("--rf-trees", type=int, default=500)
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
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["doy"] = df["date"].dt.dayofyear
    df["season"] = df["month"].map(month_to_season)
    return df


def build_city_daily(daily: pd.DataFrame, threshold: float) -> pd.DataFrame:
    city = (
        daily.groupby("date", as_index=False)
        .agg(
            city_mean_mda8=("mda8_o3", "mean"),
            city_median_mda8=("mda8_o3", "median"),
            city_max_mda8=("mda8_o3", "max"),
            affected_sites=("mda8_o3", lambda x: x.notna().sum()),
            event_site_count=("mda8_o3", lambda x: (x >= threshold).sum()),
            temp_mean=("temp_mean", "mean"),
            temp_max=("temp_max", "mean"),
            rh_mean=("rh_mean", "mean"),
            wind_speed_mean=("wind_speed_mean", "mean"),
            wind_dir_sin=("wind_dir_sin", "mean"),
            wind_dir_cos=("wind_dir_cos", "mean"),
            rain_sum=("rain_sum", "mean"),
            sunshine_hours=("sunshine_hours", "mean"),
            no2_mean=("no2_mean", "mean"),
            nox_mean=("nox_mean", "mean"),
            co_mean=("co_mean", "mean"),
            pm25_mean=("pm25_mean", "mean"),
        )
        .dropna(subset=["city_max_mda8"])
    )
    city["year"] = city["date"].dt.year
    city["month"] = city["date"].dt.month
    city["doy"] = city["date"].dt.dayofyear
    city["season"] = city["month"].map(month_to_season)
    city["rain_log1p"] = np.log1p(city["rain_sum"].clip(lower=0))
    city["doy_sin"] = np.sin(2 * np.pi * city["doy"] / 365.25)
    city["doy_cos"] = np.cos(2 * np.pi * city["doy"] / 365.25)
    city["year_index"] = city["year"] - city["year"].min()
    city["high_o3_event_60ppb"] = city["city_max_mda8"] >= threshold
    city["event_threshold_ppb"] = threshold
    return city


def write_fixed_threshold_event_tables(
    daily: pd.DataFrame,
    city: pd.DataFrame,
    threshold: float,
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    daily_labeled = daily.copy()
    daily_labeled["site_high_o3_60ppb"] = daily_labeled["mda8_o3"] >= threshold
    daily_labeled["event_threshold_ppb"] = threshold
    daily_labeled[
        [
            "site",
            "date",
            "year",
            "month",
            "season",
            "mda8_o3",
            "site_high_o3_60ppb",
            "event_threshold_ppb",
            "temp_max",
            "rh_mean",
            "wind_speed_mean",
            "rain_sum",
            "sunshine_hours",
        ]
    ].to_csv(tables_dir / "event60_site_day_labels.csv", index=False, encoding="utf-8-sig")

    city.to_csv(tables_dir / "event60_city_daily_dataset.csv", index=False, encoding="utf-8-sig")
    events = city[city["high_o3_event_60ppb"]].copy()
    events.to_csv(tables_dir / "event60_city_event_days.csv", index=False, encoding="utf-8-sig")

    by_month = (
        events.groupby("month", as_index=False)
        .size()
        .rename(columns={"size": "n_event_days"})
        .sort_values("month")
    )
    by_season = (
        events.groupby("season", as_index=False)
        .size()
        .rename(columns={"size": "n_event_days"})
        .sort_values("season")
    )
    by_year = (
        city.groupby("year", as_index=False)
        .agg(
            total_days=("date", "count"),
            event_days=("high_o3_event_60ppb", "sum"),
            mean_city_max_mda8=("city_max_mda8", "mean"),
            p90_city_max_mda8=("city_max_mda8", lambda x: x.quantile(0.90)),
        )
    )
    by_year["event_fraction"] = by_year["event_days"] / by_year["total_days"]
    by_month.to_csv(tables_dir / "event60_counts_by_month.csv", index=False, encoding="utf-8-sig")
    by_season.to_csv(tables_dir / "event60_counts_by_season.csv", index=False, encoding="utf-8-sig")
    by_year.to_csv(tables_dir / "event60_counts_by_year.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for feature in EVENT_SUMMARY_FEATURES:
        for is_event, values in city.groupby("high_o3_event_60ppb")[feature]:
            summary_rows.append(
                {
                    "feature": feature,
                    "period": "event" if is_event else "non_event",
                    "mean": values.mean(),
                    "median": values.median(),
                    "p25": values.quantile(0.25),
                    "p75": values.quantile(0.75),
                    "n": values.count(),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        tables_dir / "event60_meteorology_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    anomaly_rows = []
    for feature in EVENT_SUMMARY_FEATURES:
        climatology = city.groupby("month")[feature].transform("mean")
        anomaly = city[feature] - climatology
        tmp = pd.DataFrame({"event": city["high_o3_event_60ppb"], "anomaly": anomaly})
        for is_event, values in tmp.groupby("event")["anomaly"]:
            anomaly_rows.append(
                {
                    "feature": feature,
                    "period": "event" if is_event else "non_event",
                    "monthly_adjusted_mean": values.mean(),
                    "monthly_adjusted_median": values.median(),
                    "monthly_adjusted_p25": values.quantile(0.25),
                    "monthly_adjusted_p75": values.quantile(0.75),
                    "n": values.count(),
                }
            )
    pd.DataFrame(anomaly_rows).to_csv(
        tables_dir / "event60_meteorology_month_adjusted_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= 2023].copy()
    test = df[df["year"] >= 2024].copy()
    return train, test


def evaluate_classifier(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def fit_logistic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features])
    x_test = scaler.transform(test[features])
    y_train = train["high_o3_event_60ppb"].astype(int).to_numpy()
    y_test = test["high_o3_event_60ppb"].astype(int).to_numpy()

    model = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)
    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    result = evaluate_classifier(y_test, y_prob, y_pred)

    coef = pd.DataFrame(
        {
            "feature": features,
            "coefficient_per_sd": model.coef_[0],
            "odds_ratio_per_sd": np.exp(model.coef_[0]),
        }
    )
    coef = add_statsmodels_logistic_ci(train, features, scaler, coef)

    cm = confusion_frame(y_test, y_pred, "Logistic regression")
    roc = roc_frame(y_test, y_prob, "Logistic regression")
    return result, coef, cm, roc


def add_statsmodels_logistic_ci(
    train: pd.DataFrame,
    features: list[str],
    scaler: StandardScaler,
    coef: pd.DataFrame,
) -> pd.DataFrame:
    if sm is None:
        coef["p_value"] = np.nan
        coef["odds_ratio_ci_low"] = np.nan
        coef["odds_ratio_ci_high"] = np.nan
        return coef.sort_values("odds_ratio_per_sd", ascending=False)

    x_train = pd.DataFrame(scaler.transform(train[features]), columns=features, index=train.index)
    x_train = sm.add_constant(x_train, has_constant="add")
    y_train = train["high_o3_event_60ppb"].astype(int)
    try:
        fit = sm.Logit(y_train, x_train).fit(disp=False, maxiter=300)
        params = fit.params.drop("const", errors="ignore")
        conf = fit.conf_int().drop("const", errors="ignore")
        stats = pd.DataFrame(
            {
                "feature": params.index,
                "statsmodels_coefficient": params.to_numpy(),
                "p_value": fit.pvalues.drop("const", errors="ignore").to_numpy(),
                "odds_ratio_ci_low": np.exp(conf[0].to_numpy()),
                "odds_ratio_ci_high": np.exp(conf[1].to_numpy()),
            }
        )
        coef = coef.merge(stats, on="feature", how="left")
    except Exception:
        coef["statsmodels_coefficient"] = np.nan
        coef["p_value"] = np.nan
        coef["odds_ratio_ci_low"] = np.nan
        coef["odds_ratio_ci_high"] = np.nan
    return coef.sort_values("odds_ratio_per_sd", ascending=False)


def fit_random_forest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    n_estimators: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train = train[features].astype(float)
    x_test = test[features].astype(float)
    y_train = train["high_o3_event_60ppb"].astype(int).to_numpy()
    y_test = test["high_o3_event_60ppb"].astype(int).to_numpy()

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    result = evaluate_classifier(y_test, y_prob, y_pred)

    perm = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_mean_auc": perm.importances_mean,
            "importance_sd_auc": perm.importances_std,
            "gini_importance": model.feature_importances_,
        }
    ).sort_values("importance_mean_auc", ascending=False)

    cm = confusion_frame(y_test, y_pred, "Random Forest")
    roc = roc_frame(y_test, y_prob, "Random Forest")
    return result, importance, cm, roc


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray, model: str) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    rows = []
    for actual_idx, actual in enumerate(["non_event", "event"]):
        for pred_idx, predicted in enumerate(["non_event", "event"]):
            rows.append(
                {
                    "model": model,
                    "actual": actual,
                    "predicted": predicted,
                    "n": int(cm[actual_idx, pred_idx]),
                }
            )
    return pd.DataFrame(rows)


def roc_frame(y_true: np.ndarray, y_prob: np.ndarray, model: str) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame({"model": model, "fpr": fpr, "tpr": tpr, "threshold": thresholds})


def run_classification(city: pd.DataFrame, tables_dir: Path, figures_dir: Path, rf_trees: int) -> None:
    configs = [
        ("meteorology", BASE_FEATURES),
        ("meteorology_sunshine", SUNSHINE_FEATURES),
        ("meteorology_sunshine_precursors", PRECURSOR_FEATURES),
    ]
    performance_rows = []
    coef_frames = []
    importance_frames = []
    cm_frames = []
    roc_frames = []

    for experiment, features in configs:
        data = city.dropna(subset=["high_o3_event_60ppb"] + features).copy()
        train, test = temporal_split(data)
        train_event_rate = train["high_o3_event_60ppb"].mean()
        test_event_rate = test["high_o3_event_60ppb"].mean()
        print(
            f"{experiment}: rows={len(data)}, train={train['year'].min()}-{train['year'].max()} "
            f"(event rate={train_event_rate:.3f}), test={test['year'].min()}-{test['year'].max()} "
            f"(event rate={test_event_rate:.3f})",
            flush=True,
        )

        logit_result, coef, cm, roc = fit_logistic(train, test, features)
        logit_result.update(
            {
                "experiment": experiment,
                "model": "Logistic regression",
                "n_train": len(train),
                "n_test": len(test),
                "train_years": f"{train['year'].min()}-{train['year'].max()}",
                "test_years": f"{test['year'].min()}-{test['year'].max()}",
                "train_event_rate": train_event_rate,
                "test_event_rate": test_event_rate,
                "features": ",".join(features),
            }
        )
        coef["experiment"] = experiment
        performance_rows.append(logit_result)
        coef_frames.append(coef)
        cm["experiment"] = experiment
        roc["experiment"] = experiment
        cm_frames.append(cm)
        roc_frames.append(roc)

        rf_result, importance, cm, roc = fit_random_forest(train, test, features, rf_trees)
        rf_result.update(
            {
                "experiment": experiment,
                "model": "Random Forest",
                "n_train": len(train),
                "n_test": len(test),
                "train_years": f"{train['year'].min()}-{train['year'].max()}",
                "test_years": f"{test['year'].min()}-{test['year'].max()}",
                "train_event_rate": train_event_rate,
                "test_event_rate": test_event_rate,
                "features": ",".join(features),
            }
        )
        importance["experiment"] = experiment
        performance_rows.append(rf_result)
        importance_frames.append(importance)
        cm["experiment"] = experiment
        roc["experiment"] = experiment
        cm_frames.append(cm)
        roc_frames.append(roc)

    performance = pd.DataFrame(performance_rows)
    ordered_cols = [
        "experiment",
        "model",
        "n_train",
        "n_test",
        "train_years",
        "test_years",
        "train_event_rate",
        "test_event_rate",
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "features",
    ]
    performance[ordered_cols].to_csv(
        tables_dir / "event60_classification_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(coef_frames, ignore_index=True).to_csv(
        tables_dir / "event60_logistic_odds_ratios.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(importance_frames, ignore_index=True).to_csv(
        tables_dir / "event60_rf_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(cm_frames, ignore_index=True).to_csv(
        tables_dir / "event60_classification_confusion_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(roc_frames, ignore_index=True).to_csv(
        tables_dir / "event60_classification_roc_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_classification_figures(performance, coef_frames, importance_frames, roc_frames, cm_frames, figures_dir)


def write_classification_figures(
    performance: pd.DataFrame,
    coef_frames: list[pd.DataFrame],
    importance_frames: list[pd.DataFrame],
    roc_frames: list[pd.DataFrame],
    cm_frames: list[pd.DataFrame],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    roc = pd.concat(roc_frames, ignore_index=True)
    label_map = {
        "meteorology": "Met",
        "meteorology_sunshine": "Met+Sun",
        "meteorology_sunshine_precursors": "Met+Sun+Prec",
        "Logistic regression": "LR",
        "Random Forest": "RF",
    }

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for (experiment, model), group in roc.groupby(["experiment", "model"]):
        auc = performance.loc[
            (performance["experiment"] == experiment) & (performance["model"] == model),
            "roc_auc",
        ].iloc[0]
        ax.plot(group["fpr"], group["tpr"], label=f"{label_map[experiment]} {label_map[model]} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("MDA8 O3 >= 60 ppb event classification")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "event60_classification_roc_curve.png", dpi=300)
    plt.close(fig)

    cm = pd.concat(cm_frames, ignore_index=True)
    best_exp = performance.sort_values("roc_auc", ascending=False).iloc[0]["experiment"]
    cm_plot = cm[cm["experiment"] == best_exp].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), squeeze=False)
    for ax, model in zip(axes.ravel(), ["Logistic regression", "Random Forest"]):
        pivot = (
            cm_plot[cm_plot["model"] == model]
            .pivot(index="actual", columns="predicted", values="n")
            .reindex(index=["non_event", "event"], columns=["non_event", "event"])
        )
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{model}\n{best_exp}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(figures_dir / "event60_classification_confusion_matrix.png", dpi=300)
    plt.close(fig)

    coef = pd.concat(coef_frames, ignore_index=True)
    coef_plot = coef[coef["experiment"] == "meteorology_sunshine"].copy()
    coef_plot["distance"] = (coef_plot["odds_ratio_per_sd"] - 1).abs()
    coef_plot = coef_plot.sort_values("distance", ascending=False).head(10).sort_values("odds_ratio_per_sd")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(coef_plot["feature"], coef_plot["odds_ratio_per_sd"], color="#7b9e87")
    ax.axvline(1.0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Odds ratio per 1 SD increase")
    ax.set_title("Logistic regression odds ratios")
    fig.tight_layout()
    fig.savefig(figures_dir / "event60_logistic_odds_ratios.png", dpi=300)
    plt.close(fig)

    importance = pd.concat(importance_frames, ignore_index=True)
    imp_plot = importance[importance["experiment"] == "meteorology_sunshine"].head(10).sort_values("importance_mean_auc")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(imp_plot["feature"], imp_plot["importance_mean_auc"], color="#2b7a9b")
    ax.set_xlabel("Permutation importance (AUC decrease)")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(figures_dir / "event60_rf_feature_importance.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    daily = load_daily(Path(args.daily))
    city = build_city_daily(daily, args.threshold)
    write_fixed_threshold_event_tables(daily, city, args.threshold, tables_dir)
    run_classification(city, tables_dir, figures_dir, args.rf_trees)

    n_events = int(city["high_o3_event_60ppb"].sum())
    print(
        f"Fixed event threshold MDA8 O3 >= {args.threshold:.1f} ppb: "
        f"{n_events}/{len(city)} city event days ({n_events / len(city):.3f}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
