from __future__ import annotations

import argparse
from dataclasses import dataclass
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


DEFAULT_OUT = Path("analysis_outputs/revision/T2_T3_models")

OLD_FEATURES = [
    "temp_max",
    "rh_mean",
    "wind_speed_mean",
    "rain_log1p",
    "wind_dir_sin",
    "wind_dir_cos",
    "doy",
    "year_index",
]

NEW_RF_FEATURES = [
    "temp_max",
    "rh_mean",
    "wind_speed_mean",
    "rain_log1p",
    "day_wind_u",
    "day_wind_v",
    "doy_sin",
    "doy_cos",
]

NEW_GAM_FEATURES = [
    "temp_max",
    "rh_mean",
    "wind_speed_mean",
    "rain_log1p",
    "day_wind_u",
    "day_wind_v",
    "doy",
]


@dataclass(frozen=True)
class ModelExperiment:
    experiment: str
    level: str
    target: str
    rf_features: list[str]
    gam_features: list[str]
    use_site: bool
    cyclic_doy: bool
    has_year_index: bool
    wind_spec: str
    doy_spec: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run revision T2/T3 models: day/night wind features plus no-year/cyclic-DOY model comparison."
    )
    parser.add_argument("--daily", default="outputs/tables/daily_mda8_o3_2018_2025.csv")
    parser.add_argument("--hourly", default="outputs/tables/hourly_o3_meteo_2018_2025.csv")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--perm-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def month_to_season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def vector_wind(direction_deg: pd.Series, speed: pd.Series) -> tuple[pd.Series, pd.Series]:
    theta = np.deg2rad(direction_deg.astype(float))
    u = -speed.astype(float) * np.sin(theta)
    v = -speed.astype(float) * np.cos(theta)
    return u, v


def direction_from_uv(u: pd.Series, v: pd.Series) -> pd.Series:
    direction = (np.rad2deg(np.arctan2(-u.astype(float), -v.astype(float))) + 360.0) % 360.0
    return direction


def build_period_wind(hourly: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly.copy()
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    hourly["date"] = pd.to_datetime(hourly["date"])
    hourly = ensure_numeric(hourly, ["hour", "WIND_DIREC", "WIND_SPEED"])
    hourly = hourly.dropna(subset=["site", "date", "hour", "WIND_DIREC", "WIND_SPEED"]).copy()
    hourly = hourly[
        hourly["WIND_DIREC"].between(0, 360, inclusive="both")
        & hourly["WIND_SPEED"].ge(0)
        & hourly["WIND_SPEED"].le(30)
    ].copy()

    hourly["wind_u"], hourly["wind_v"] = vector_wind(hourly["WIND_DIREC"], hourly["WIND_SPEED"])
    hourly["period"] = np.select(
        [
            hourly["hour"].between(10, 16, inclusive="both"),
            hourly["hour"].ge(20) | hourly["hour"].le(6),
        ],
        ["day", "night"],
        default="other",
    )
    hourly = hourly[hourly["period"].isin(["day", "night"])].copy()

    grouped = (
        hourly.groupby(["site", "date", "period"], observed=True)
        .agg(
            wind_u=("wind_u", "mean"),
            wind_v=("wind_v", "mean"),
            wind_speed_scalar=("WIND_SPEED", "mean"),
            wind_valid_hours=("WIND_SPEED", "count"),
        )
        .reset_index()
    )
    grouped["wind_speed_vector"] = np.sqrt(grouped["wind_u"] ** 2 + grouped["wind_v"] ** 2)
    grouped["wind_dir_vector"] = direction_from_uv(grouped["wind_u"], grouped["wind_v"])

    wide = grouped.pivot(index=["site", "date"], columns="period")
    wide.columns = [f"{period}_{name}" for name, period in wide.columns]
    wide = wide.reset_index()
    return wide


def prepare_station_day(daily_path: Path, hourly_path: Path, out_tables: Path) -> pd.DataFrame:
    daily = pd.read_csv(daily_path, parse_dates=["date"])
    num_cols = [
        "mda8_o3",
        "temp_max",
        "rh_mean",
        "wind_speed_mean",
        "rain_sum",
        "wind_dir_sin",
        "wind_dir_cos",
        "doy",
        "year",
        "month",
        "sunshine_hours",
    ]
    daily = ensure_numeric(daily, num_cols)
    daily["rain_log1p"] = np.log1p(daily["rain_sum"].clip(lower=0))
    daily["year_index"] = daily["year"] - daily["year"].min()
    daily["doy_sin"] = np.sin(2.0 * np.pi * daily["doy"] / 366.0)
    daily["doy_cos"] = np.cos(2.0 * np.pi * daily["doy"] / 366.0)
    daily["season"] = daily["month"].map(month_to_season)

    hourly = pd.read_csv(hourly_path, parse_dates=["date"])
    wind = build_period_wind(hourly)
    merged = daily.merge(wind, on=["site", "date"], how="left", validate="one_to_one")
    merged.to_csv(out_tables / "T3_daily_mda8_o3_with_daynight_wind.csv", index=False, encoding="utf-8-sig")

    availability_cols = [
        "day_wind_u",
        "day_wind_v",
        "day_wind_valid_hours",
        "night_wind_u",
        "night_wind_v",
        "night_wind_valid_hours",
    ]
    rows = []
    for col in availability_cols:
        if col in merged.columns:
            rows.append(
                {
                    "feature": col,
                    "n_non_missing": int(merged[col].notna().sum()),
                    "n_rows": int(len(merged)),
                    "coverage_pct": float(100.0 * merged[col].notna().mean()),
                }
            )
    pd.DataFrame(rows).to_csv(out_tables / "T3_daynight_wind_feature_coverage.csv", index=False, encoding="utf-8-sig")
    return merged


def build_city_day(station_day: pd.DataFrame) -> pd.DataFrame:
    agg_map = {
        "mda8_o3": ["mean", "max", "count"],
        "temp_max": "mean",
        "rh_mean": "mean",
        "wind_speed_mean": "mean",
        "rain_sum": "mean",
        "rain_log1p": "mean",
        "wind_dir_sin": "mean",
        "wind_dir_cos": "mean",
        "day_wind_u": "mean",
        "day_wind_v": "mean",
        "night_wind_u": "mean",
        "night_wind_v": "mean",
        "sunshine_hours": "first",
        "year": "first",
        "month": "first",
        "doy": "first",
        "doy_sin": "first",
        "doy_cos": "first",
    }
    city = station_day.groupby("date", as_index=False).agg(agg_map)
    city.columns = [
        "_".join([str(part) for part in col if part]) if isinstance(col, tuple) else col
        for col in city.columns
    ]
    city = city.rename(
        columns={
            "mda8_o3_mean": "mda8_o3",
            "mda8_o3_max": "city_max_mda8",
            "mda8_o3_count": "n_sites",
            "temp_max_mean": "temp_max",
            "rh_mean_mean": "rh_mean",
            "wind_speed_mean_mean": "wind_speed_mean",
            "rain_sum_mean": "rain_sum",
            "rain_log1p_mean": "rain_log1p",
            "wind_dir_sin_mean": "wind_dir_sin",
            "wind_dir_cos_mean": "wind_dir_cos",
            "day_wind_u_mean": "day_wind_u",
            "day_wind_v_mean": "day_wind_v",
            "night_wind_u_mean": "night_wind_u",
            "night_wind_v_mean": "night_wind_v",
            "sunshine_hours_first": "sunshine_hours",
            "year_first": "year",
            "month_first": "month",
            "doy_first": "doy",
            "doy_sin_first": "doy_sin",
            "doy_cos_first": "doy_cos",
        }
    )
    city["year_index"] = city["year"] - city["year"].min()
    city["season"] = city["month"].map(month_to_season)
    return city


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(df["year"].dropna().unique())
    split_year = years[-2] if len(years) >= 4 else years[int(len(years) * 0.8)]
    train = df[df["year"] < split_year].copy()
    test = df[df["year"] >= split_year].copy()
    return train, test


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def rf_design(train: pd.DataFrame, test: pd.DataFrame, features: list[str], use_site: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = features + (["site"] if use_site else [])
    X_train = train[cols].copy()
    X_test = test[cols].copy()
    if use_site:
        X_train = pd.get_dummies(X_train, columns=["site"], drop_first=False)
        X_test = pd.get_dummies(X_test, columns=["site"], drop_first=False)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train.astype(float), X_test.astype(float)


def group_feature_name(feature: str) -> str:
    if feature.startswith("site_"):
        return "site"
    return feature


def fit_rf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exp: ModelExperiment,
    rf_trees: int,
    perm_repeats: int,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    X_train, X_test = rf_design(train, test, exp.rf_features, exp.use_site)
    y_train = train[exp.target].to_numpy(float)
    y_test = test[exp.target].to_numpy(float)

    model = RandomForestRegressor(
        n_estimators=rf_trees,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    row = build_performance_row(exp, "Random Forest", train, test, metrics(y_test, pred))
    row["oob_r2"] = float(getattr(model, "oob_score_", np.nan))
    row["gam_lam"] = ""

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=perm_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    imp = pd.DataFrame(
        {
            "experiment": exp.experiment,
            "level": exp.level,
            "model": "Random Forest",
            "feature": X_test.columns,
            "feature_group": [group_feature_name(c) for c in X_test.columns],
            "importance_mean": importance.importances_mean,
            "importance_sd": importance.importances_std,
        }
    ).sort_values(["experiment", "importance_mean"], ascending=[True, False])
    return row, imp


def make_gam_terms(features: list[str], use_site: bool, cyclic_doy: bool):
    terms = None
    for idx, feature in enumerate(features):
        if feature == "doy" and cyclic_doy:
            term = s(idx, n_splines=12, basis="cp", edge_knots=[0.5, 366.5])
        else:
            term = s(idx, n_splines=10)
        terms = term if terms is None else terms + term
    if use_site:
        terms += f(len(features))
    return terms


def add_site_code(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    categories = sorted(train["site"].dropna().unique())
    site_map = {site: code for code, site in enumerate(categories)}
    train["site_code"] = train["site"].map(site_map).astype(float)
    test["site_code"] = test["site"].map(site_map).astype(float)
    return train, test


def fit_gam(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exp: ModelExperiment,
) -> tuple[dict[str, object], LinearGAM, pd.DataFrame, pd.DataFrame, list[str]]:
    if exp.use_site:
        train, test = add_site_code(train, test)
    gam_features = exp.gam_features + (["site_code"] if exp.use_site else [])
    X_train = train[gam_features].to_numpy(float)
    X_test = test[gam_features].to_numpy(float)
    y_train = train[exp.target].to_numpy(float)
    y_test = test[exp.target].to_numpy(float)

    terms = make_gam_terms(exp.gam_features, exp.use_site, exp.cyclic_doy)
    gam = LinearGAM(terms)
    gam.gridsearch(X_train, y_train, lam=np.logspace(-2, 2, 5), progress=False)
    pred = gam.predict(X_test)

    row = build_performance_row(exp, "GAM", train, test, metrics(y_test, pred))
    row["oob_r2"] = ""
    row["gam_lam"] = ";".join([f"{x:.4g}" for x in np.ravel(gam.lam)])
    return row, gam, train, test, gam_features


def build_performance_row(
    exp: ModelExperiment,
    model: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    values: dict[str, float],
) -> dict[str, object]:
    return {
        "experiment": exp.experiment,
        "level": exp.level,
        "model": model,
        "target": exp.target,
        "wind_spec": exp.wind_spec,
        "doy_spec": exp.doy_spec,
        "has_year_index": exp.has_year_index,
        "features_rf": ",".join(exp.rf_features),
        "features_gam": ",".join(exp.gam_features),
        "n_total": int(len(train) + len(test)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_years": f"{int(train['year'].min())}-{int(train['year'].max())}",
        "test_years": f"{int(test['year'].min())}-{int(test['year'].max())}",
        **values,
    }


def gam_reference_frame(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    ref = {}
    for col in features:
        if col == "site_code":
            ref[col] = float(train[col].mode().iloc[0])
        else:
            ref[col] = float(train[col].median())
    return pd.DataFrame([ref])


def extract_gam_partial_effects(
    gam: LinearGAM,
    train: pd.DataFrame,
    gam_features: list[str],
    exp: ModelExperiment,
) -> pd.DataFrame:
    rows = []
    ref = gam_reference_frame(train, gam_features)
    for idx, feature in enumerate(exp.gam_features):
        if feature == "doy":
            grid = np.arange(1, 367, dtype=float)
        else:
            lo = train[feature].quantile(0.01)
            hi = train[feature].quantile(0.99)
            grid = np.linspace(lo, hi, 100)
        X = pd.concat([ref] * len(grid), ignore_index=True)
        X[feature] = grid
        effect = gam.partial_dependence(term=idx, X=X[gam_features].to_numpy(float))
        rows.append(
            pd.DataFrame(
                {
                    "experiment": exp.experiment,
                    "level": exp.level,
                    "feature": feature,
                    "value": grid,
                    "partial_effect": effect,
                    "cyclic_doy": exp.cyclic_doy,
                    "doy_spec": exp.doy_spec,
                    "wind_spec": exp.wind_spec,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def get_experiments() -> list[ModelExperiment]:
    return [
        ModelExperiment(
            experiment="old_station_day_dailywind_year_nocyclic",
            level="station_day",
            target="mda8_o3",
            rf_features=OLD_FEATURES,
            gam_features=OLD_FEATURES,
            use_site=True,
            cyclic_doy=False,
            has_year_index=True,
            wind_spec="daily mean wind direction sin/cos",
            doy_spec="raw doy spline; RF raw doy",
        ),
        ModelExperiment(
            experiment="new_station_day_daywind_no_year_cyclic",
            level="station_day",
            target="mda8_o3",
            rf_features=NEW_RF_FEATURES,
            gam_features=NEW_GAM_FEATURES,
            use_site=True,
            cyclic_doy=True,
            has_year_index=False,
            wind_spec="daytime 10-16 LST vector wind u/v",
            doy_spec="RF doy sin/cos; GAM cyclic doy",
        ),
        ModelExperiment(
            experiment="old_city_day_dailywind_year_nocyclic",
            level="city_day",
            target="mda8_o3",
            rf_features=OLD_FEATURES,
            gam_features=OLD_FEATURES,
            use_site=False,
            cyclic_doy=False,
            has_year_index=True,
            wind_spec="city mean daily wind direction sin/cos",
            doy_spec="raw doy spline; RF raw doy",
        ),
        ModelExperiment(
            experiment="new_city_day_daywind_no_year_cyclic",
            level="city_day",
            target="mda8_o3",
            rf_features=NEW_RF_FEATURES,
            gam_features=NEW_GAM_FEATURES,
            use_site=False,
            cyclic_doy=True,
            has_year_index=False,
            wind_spec="city mean daytime 10-16 LST vector wind u/v",
            doy_spec="RF doy sin/cos; GAM cyclic doy",
        ),
    ]


def fit_all_models(
    station_day: pd.DataFrame,
    city_day: pd.DataFrame,
    out_tables: Path,
    out_figures: Path,
    rf_trees: int,
    perm_repeats: int,
    random_state: int,
) -> None:
    performance_rows: list[dict[str, object]] = []
    importance_frames: list[pd.DataFrame] = []
    partial_frames: list[pd.DataFrame] = []

    datasets = {"station_day": station_day, "city_day": city_day}
    for exp in get_experiments():
        data = datasets[exp.level].copy()
        required = sorted(set([exp.target] + exp.rf_features + exp.gam_features + ["year"] + (["site"] if exp.use_site else [])))
        data = data.dropna(subset=required).copy()
        train, test = temporal_split(data)
        print(
            f"[T2/T3] {exp.experiment}: n={len(data)}, train={train['year'].min()}-{train['year'].max()}, "
            f"test={test['year'].min()}-{test['year'].max()}",
            flush=True,
        )

        rf_row, rf_imp = fit_rf(train, test, exp, rf_trees, perm_repeats, random_state)
        gam_row, gam, gam_train, _, gam_features = fit_gam(train, test, exp)
        performance_rows.extend([rf_row, gam_row])
        importance_frames.append(rf_imp)
        partial_frames.append(extract_gam_partial_effects(gam, gam_train, gam_features, exp))

    performance = pd.DataFrame(performance_rows)
    performance.to_csv(out_tables / "T2_T3_model_performance.csv", index=False, encoding="utf-8-sig")

    importance = pd.concat(importance_frames, ignore_index=True)
    importance.to_csv(out_tables / "T2_T3_rf_permutation_importance.csv", index=False, encoding="utf-8-sig")
    grouped = (
        importance.groupby(["experiment", "level", "model", "feature_group"], as_index=False)
        .agg(
            importance_mean=("importance_mean", "sum"),
            importance_sd=("importance_sd", lambda x: float(np.sqrt(np.sum(np.square(x))))),
        )
        .sort_values(["experiment", "importance_mean"], ascending=[True, False])
    )
    grouped.to_csv(out_tables / "T2_T3_rf_permutation_importance_grouped.csv", index=False, encoding="utf-8-sig")

    partials = pd.concat(partial_frames, ignore_index=True)
    partials.to_csv(out_tables / "T2_T3_gam_partial_effects.csv", index=False, encoding="utf-8-sig")
    make_figures(performance, grouped, partials, out_figures)


def make_figures(performance: pd.DataFrame, grouped: pd.DataFrame, partials: pd.DataFrame, out_figures: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    plot_perf = performance.copy()
    plot_perf["spec"] = np.where(plot_perf["has_year_index"].astype(bool), "old", "new")
    plot_perf["label"] = plot_perf["level"] + "\n" + plot_perf["spec"] + "\n" + plot_perf["model"]
    sns.barplot(data=plot_perf, x="label", y="r2", hue="level", dodge=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Test R2")
    ax.set_title("T2/T3 model performance comparison")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_figures / "T2_T3_model_performance_r2.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=True)
    for ax, level in zip(axes, ["station_day", "city_day"]):
        sub = partials[(partials["feature"] == "doy") & (partials["level"] == level)].copy()
        sub["spec"] = np.where(sub["cyclic_doy"], "new cyclic doy", "old non-cyclic doy")
        sns.lineplot(data=sub, x="value", y="partial_effect", hue="spec", ax=ax)
        ax.set_title(f"GAM DOY partial effect: {level}")
        ax.set_xlabel("Day of year")
        ax.set_ylabel("Partial effect")
    fig.tight_layout()
    fig.savefig(out_figures / "T2_T3_gam_doy_cyclic_comparison.png", dpi=220)
    plt.close(fig)

    top = (
        grouped[grouped["experiment"].str.startswith("new_")]
        .sort_values(["experiment", "importance_mean"], ascending=[True, False])
        .groupby("experiment", as_index=False)
        .head(8)
        .copy()
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sns.barplot(data=top, y="feature_group", x="importance_mean", hue="experiment", ax=ax)
    ax.set_xlabel("Permutation importance, grouped")
    ax.set_ylabel("")
    ax.set_title("Top RF predictors in revised wind/cyclic models")
    fig.tight_layout()
    fig.savefig(out_figures / "T2_T3_new_rf_top_importance.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_tables = out_dir / "tables"
    out_figures = out_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    station_day = prepare_station_day(Path(args.daily), Path(args.hourly), out_tables)
    city_day = build_city_day(station_day)
    city_day.to_csv(out_tables / "T2_city_day_model_dataset.csv", index=False, encoding="utf-8-sig")

    fit_all_models(
        station_day=station_day,
        city_day=city_day,
        out_tables=out_tables,
        out_figures=out_figures,
        rf_trees=args.rf_trees,
        perm_repeats=args.perm_repeats,
        random_state=args.random_state,
    )
    print(f"[T2/T3] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
