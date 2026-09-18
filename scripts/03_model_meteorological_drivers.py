from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit GAM and Random Forest models for MDA8 O3.")
    parser.add_argument("--daily", default="outputs/tables/daily_mda8_o3_2018_2025.csv")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument(
        "--include-sunshine",
        action="store_true",
        help="Include sunshine_hours when the daily table contains CODiS sunshine data.",
    )
    return parser.parse_args()


def prepare_model_data(path: Path, include_sunshine: bool) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["mda8_o3"]).copy()
    df["rain_log1p"] = np.log1p(df["rain_sum"].clip(lower=0))
    df["year_index"] = df["year"] - df["year"].min()
    features = BASE_FEATURES.copy()
    if include_sunshine:
        features.append("sunshine_hours")
    df = df.dropna(subset=features + ["site"]).copy()
    df["site_code"] = df["site"].astype("category").cat.codes
    return df, features


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


def metrics(y_true: np.ndarray, y_pred: np.ndarray, model: str, n_train: int, n_test: int) -> dict[str, float | int | str]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model,
        "n_train": n_train,
        "n_test": n_test,
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def fit_random_forest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    tables_dir: Path,
    figures_dir: Path,
    suffix: str,
) -> dict[str, float | int | str]:
    rf_features = features + ["site"]
    X_train = pd.get_dummies(train[rf_features], columns=["site"], drop_first=False)
    X_test = pd.get_dummies(test[rf_features], columns=["site"], drop_first=False)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    y_train = train["mda8_o3"].to_numpy()
    y_test = test["mda8_o3"].to_numpy()

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    imp = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_sd": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp.to_csv(tables_dir / f"rf_permutation_importance{suffix}.csv", index=False, encoding="utf-8-sig")

    top = [feature for feature in imp["feature"].head(6) if not feature.startswith("site_")]
    if top:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        PartialDependenceDisplay.from_estimator(model, X_train, features=top[:4], ax=ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"rf_partial_dependence_top_features{suffix}.png", dpi=220)
        plt.close(fig)

    result = metrics(y_test, pred, "Random Forest", len(train), len(test))
    result["oob_r2"] = getattr(model, "oob_score_", np.nan)
    return result


def fit_gam(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    tables_dir: Path,
    figures_dir: Path,
    suffix: str,
) -> dict[str, float | int | str]:
    gam_features = features + ["site_code"]
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

    rows: list[pd.DataFrame] = []
    for idx, feature in enumerate(features):
        grid = np.linspace(train[feature].quantile(0.01), train[feature].quantile(0.99), 100)
        reference = train[gam_features].median(numeric_only=True).to_frame().T
        repeated = pd.concat([reference] * len(grid), ignore_index=True)
        repeated[feature] = grid
        pdp = gam.partial_dependence(term=idx, X=repeated[gam_features].to_numpy())
        rows.append(pd.DataFrame({"feature": feature, "value": grid, "partial_effect": pdp}))

    effects = pd.concat(rows, ignore_index=True)
    effects.to_csv(tables_dir / f"gam_partial_effects{suffix}.csv", index=False, encoding="utf-8-sig")

    n_cols = 2
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.0, 2.8 * n_rows), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        subset = effects[effects["feature"] == feature]
        ax.plot(subset["value"], subset["partial_effect"], color="#1f6f8b")
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("partial effect")
    for ax in axes.ravel()[len(features) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / f"gam_partial_effects{suffix}.png", dpi=220)
    plt.close(fig)

    return metrics(y_test, pred, "GAM", len(train), len(test))


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df, features = prepare_model_data(Path(args.daily), args.include_sunshine)
    train, test = temporal_split(df)
    suffix = "_with_sunshine" if args.include_sunshine else ""
    print(
        f"Model rows: {len(df)}; train years {train['year'].min()}-{train['year'].max()}, "
        f"test years {test['year'].min()}-{test['year'].max()}",
        flush=True,
    )

    results = [
        fit_random_forest(train, test, features, tables_dir, figures_dir, suffix),
        fit_gam(train, test, features, tables_dir, figures_dir, suffix),
    ]
    pd.DataFrame(results).to_csv(tables_dir / f"model_performance{suffix}.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote model outputs to {tables_dir} and {figures_dir}", flush=True)


if __name__ == "__main__":
    main()
