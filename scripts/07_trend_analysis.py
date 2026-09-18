from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

SEASON_ORDER = ["spring", "summer", "autumn", "winter"]
SEASON_LABELS = {
    "spring": "Spring",
    "summer": "Summer",
    "autumn": "Autumn",
    "winter": "Winter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trend tests for Kaohsiung MDA8 O3.")
    parser.add_argument("--daily", default="outputs/tables/daily_mda8_o3_2018_2025.csv")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def mann_kendall(values: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    y = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(y)
    if n < 3:
        return empty_mk_result(n)

    diff = y[np.newaxis, :] - y[:, np.newaxis]
    upper = np.triu_indices(n, k=1)
    signs = np.sign(diff[upper])
    s_stat = float(np.sum(signs))

    _, tie_counts = np.unique(y, return_counts=True)
    tie_term = float(np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    z_score = z_from_s(s_stat, var_s)
    p_value = normal_two_sided_p(z_score) if np.isfinite(z_score) else np.nan
    pairs = n * (n - 1) / 2.0

    return {
        "n": n,
        "s": s_stat,
        "var_s": var_s,
        "z": z_score,
        "p_value": p_value,
        "kendall_tau": s_stat / pairs if pairs else np.nan,
        "direction": direction_from_s(s_stat),
        "trend": trend_label(s_stat, p_value, alpha),
    }


def seasonal_mann_kendall(
    data: pd.DataFrame,
    value_col: str,
    season_col: str,
    time_col: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    s_total = 0.0
    var_total = 0.0
    n_total = 0
    pairs_total = 0.0
    strata_used = 0

    for _, group in data.dropna(subset=[value_col, season_col, time_col]).groupby(season_col):
        group = group.sort_values(time_col)
        result = mann_kendall(group[value_col], alpha=alpha)
        n = int(result["n"])
        if n < 3 or not np.isfinite(result["var_s"]):
            continue
        s_total += float(result["s"])
        var_total += float(result["var_s"])
        n_total += n
        pairs_total += n * (n - 1) / 2.0
        strata_used += 1

    if strata_used == 0:
        result = empty_mk_result(n_total)
        result["n_strata"] = 0
        return result

    z_score = z_from_s(s_total, var_total)
    p_value = normal_two_sided_p(z_score) if np.isfinite(z_score) else np.nan
    return {
        "n": n_total,
        "n_strata": strata_used,
        "s": s_total,
        "var_s": var_total,
        "z": z_score,
        "p_value": p_value,
        "kendall_tau": s_total / pairs_total if pairs_total else np.nan,
        "direction": direction_from_s(s_total),
        "trend": trend_label(s_total, p_value, alpha),
    }


def empty_mk_result(n: int) -> dict[str, Any]:
    return {
        "n": n,
        "s": np.nan,
        "var_s": np.nan,
        "z": np.nan,
        "p_value": np.nan,
        "kendall_tau": np.nan,
        "direction": "insufficient data",
        "trend": "insufficient data",
    }


def z_from_s(s_stat: float, var_s: float) -> float:
    if var_s <= 0 or not np.isfinite(var_s):
        return np.nan
    if s_stat > 0:
        return (s_stat - 1.0) / math.sqrt(var_s)
    if s_stat < 0:
        return (s_stat + 1.0) / math.sqrt(var_s)
    return 0.0


def direction_from_s(s_stat: float) -> str:
    if not np.isfinite(s_stat):
        return "insufficient data"
    if s_stat > 0:
        return "increasing"
    if s_stat < 0:
        return "decreasing"
    return "flat"


def trend_label(s_stat: float, p_value: float, alpha: float) -> str:
    if not np.isfinite(p_value):
        return "insufficient data"
    if p_value >= alpha:
        return "no significant trend"
    return f"significant {direction_from_s(s_stat)}"


def sen_slope(
    data: pd.DataFrame,
    value_col: str,
    time_col: str,
    strata_col: str | None = None,
) -> float:
    slopes: list[np.ndarray] = []
    grouped = [(None, data)] if strata_col is None else data.groupby(strata_col)
    for _, group in grouped:
        group = group.dropna(subset=[value_col, time_col]).sort_values(time_col)
        y = group[value_col].to_numpy(dtype=float)
        t = group[time_col].to_numpy(dtype=float)
        n = len(group)
        if n < 2:
            continue
        pair_slopes = []
        for i in range(n - 1):
            dt = t[i + 1 :] - t[i]
            valid = dt > 0
            if np.any(valid):
                pair_slopes.append((y[i + 1 :][valid] - y[i]) / dt[valid])
        if pair_slopes:
            slopes.append(np.concatenate(pair_slopes))
    if not slopes:
        return np.nan
    return float(np.median(np.concatenate(slopes)))


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["decimal_year"] = out["year"] + (out["month"] - 0.5) / 12.0
    return out


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
    df = df.dropna(subset=["mda8_o3"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(month_to_season)
    return df


def build_aggregates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    city_daily = (
        df.groupby("date", as_index=False)
        .agg(
            mda8_o3=("mda8_o3", "mean"),
            city_max_mda8_o3=("mda8_o3", "max"),
            n_sites=("mda8_o3", "count"),
        )
    )
    city_daily = add_time_columns(city_daily)
    city_daily["season"] = city_daily["month"].map(month_to_season)

    city_monthly = (
        city_daily.groupby(["year", "month"], as_index=False)
        .agg(
            mda8_o3=("mda8_o3", "mean"),
            city_max_mda8_o3=("city_max_mda8_o3", "mean"),
            n_days=("mda8_o3", "count"),
            mean_n_sites=("n_sites", "mean"),
        )
    )
    city_monthly["date"] = pd.to_datetime(
        city_monthly["year"].astype(str) + "-" + city_monthly["month"].astype(str) + "-15"
    )
    city_monthly = add_time_columns(city_monthly)

    site_monthly = (
        df.groupby(["site", "year", "month"], as_index=False)
        .agg(mda8_o3=("mda8_o3", "mean"), n_days=("mda8_o3", "count"))
    )
    site_monthly["date"] = pd.to_datetime(
        site_monthly["year"].astype(str) + "-" + site_monthly["month"].astype(str) + "-15"
    )
    site_monthly = add_time_columns(site_monthly)

    city_annual = (
        city_daily.groupby("year", as_index=False)
        .agg(mda8_o3=("mda8_o3", "mean"), n_days=("mda8_o3", "count"), mean_n_sites=("n_sites", "mean"))
    )
    city_annual["decimal_year"] = city_annual["year"].astype(float)

    city_seasonal = (
        city_daily.groupby(["year", "season"], as_index=False)
        .agg(mda8_o3=("mda8_o3", "mean"), n_days=("mda8_o3", "count"), mean_n_sites=("n_sites", "mean"))
    )
    city_seasonal["decimal_year"] = city_seasonal["year"].astype(float)
    city_seasonal["season"] = pd.Categorical(city_seasonal["season"], SEASON_ORDER, ordered=True)
    city_seasonal = city_seasonal.sort_values(["season", "year"])

    site_seasonal = (
        df.groupby(["site", "year", "season"], as_index=False)
        .agg(mda8_o3=("mda8_o3", "mean"), n_days=("mda8_o3", "count"))
    )
    site_seasonal["decimal_year"] = site_seasonal["year"].astype(float)
    site_seasonal["season"] = pd.Categorical(site_seasonal["season"], SEASON_ORDER, ordered=True)
    site_seasonal = site_seasonal.sort_values(["site", "season", "year"])

    return {
        "city_daily": city_daily,
        "city_monthly": city_monthly,
        "site_monthly": site_monthly,
        "city_annual": city_annual,
        "city_seasonal": city_seasonal,
        "site_seasonal": site_seasonal,
    }


def trend_row(
    scope: str,
    metric: str,
    data: pd.DataFrame,
    mk_result: dict[str, Any],
    slope: float,
    start_year: int,
    end_year: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "scope": scope,
        "metric": metric,
        "start_year": start_year,
        "end_year": end_year,
        "n_periods": mk_result.get("n", len(data)),
        "n_strata": mk_result.get("n_strata", np.nan),
        "sen_slope_ppb_per_year": slope,
        "kendall_tau": mk_result.get("kendall_tau", np.nan),
        "s": mk_result.get("s", np.nan),
        "var_s": mk_result.get("var_s", np.nan),
        "z": mk_result.get("z", np.nan),
        "p_value": mk_result.get("p_value", np.nan),
        "direction": mk_result.get("direction", ""),
        "trend": mk_result.get("trend", ""),
    }
    if extra:
        row.update(extra)
    return row


def run_trends(aggregates: dict[str, pd.DataFrame], alpha: float) -> dict[str, pd.DataFrame]:
    city_monthly = aggregates["city_monthly"]
    site_monthly = aggregates["site_monthly"]
    city_annual = aggregates["city_annual"]
    city_seasonal = aggregates["city_seasonal"]
    site_seasonal = aggregates["site_seasonal"]
    start_year = int(city_monthly["year"].min())
    end_year = int(city_monthly["year"].max())

    city_monthly_mk = seasonal_mann_kendall(city_monthly, "mda8_o3", "month", "decimal_year", alpha=alpha)
    city_monthly_slope = sen_slope(city_monthly, "mda8_o3", "decimal_year", strata_col="month")
    city_monthly_trend = pd.DataFrame(
        [
            trend_row(
                "citywide",
                "monthly_mean_mda8_o3",
                city_monthly,
                city_monthly_mk,
                city_monthly_slope,
                start_year,
                end_year,
                {"seasonal_strata": "month"},
            )
        ]
    )

    site_rows = []
    for site, group in site_monthly.groupby("site"):
        mk_result = seasonal_mann_kendall(group, "mda8_o3", "month", "decimal_year", alpha=alpha)
        slope = sen_slope(group, "mda8_o3", "decimal_year", strata_col="month")
        site_rows.append(
            trend_row(
                "site",
                "monthly_mean_mda8_o3",
                group,
                mk_result,
                slope,
                int(group["year"].min()),
                int(group["year"].max()),
                {"site": site, "seasonal_strata": "month"},
            )
        )
    site_monthly_trend = pd.DataFrame(site_rows).sort_values("sen_slope_ppb_per_year")

    annual_mk = mann_kendall(city_annual["mda8_o3"], alpha=alpha)
    annual_slope = sen_slope(city_annual, "mda8_o3", "decimal_year")
    city_annual_trend = pd.DataFrame(
        [
            trend_row(
                "citywide",
                "annual_mean_mda8_o3",
                city_annual,
                annual_mk,
                annual_slope,
                int(city_annual["year"].min()),
                int(city_annual["year"].max()),
            )
        ]
    )

    season_rows = []
    for season in SEASON_ORDER:
        group = city_seasonal[city_seasonal["season"] == season]
        mk_result = mann_kendall(group["mda8_o3"], alpha=alpha)
        slope = sen_slope(group, "mda8_o3", "decimal_year")
        season_rows.append(
            trend_row(
                "citywide",
                "seasonal_mean_mda8_o3",
                group,
                mk_result,
                slope,
                int(group["year"].min()),
                int(group["year"].max()),
                {"season": season},
            )
        )
    city_seasonal_trend = pd.DataFrame(season_rows)

    site_season_rows = []
    for (site, season), group in site_seasonal.groupby(["site", "season"], observed=True):
        mk_result = mann_kendall(group["mda8_o3"], alpha=alpha)
        slope = sen_slope(group, "mda8_o3", "decimal_year")
        site_season_rows.append(
            trend_row(
                "site",
                "seasonal_mean_mda8_o3",
                group,
                mk_result,
                slope,
                int(group["year"].min()),
                int(group["year"].max()),
                {"site": site, "season": season},
            )
        )
    site_seasonal_trend = pd.DataFrame(site_season_rows).sort_values(["season", "sen_slope_ppb_per_year"])

    return {
        "trend_citywide_monthly_mk": city_monthly_trend,
        "trend_by_site_monthly_mk": site_monthly_trend,
        "trend_citywide_annual": city_annual_trend,
        "trend_citywide_by_season": city_seasonal_trend,
        "trend_by_site_season": site_seasonal_trend,
    }


def write_tables(trends: dict[str, pd.DataFrame], aggregates: dict[str, pd.DataFrame], tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in trends.items():
        out = tables_dir / f"{name}.csv"
        table.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Wrote {out} rows={len(table)}", flush=True)

    for name in ["city_monthly", "city_annual", "city_seasonal"]:
        out = tables_dir / f"trend_input_{name}.csv"
        aggregates[name].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Wrote {out} rows={len(aggregates[name])}", flush=True)


def plot_annual(city_annual: pd.DataFrame, trend: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    slope = float(trend.loc[0, "sen_slope_ppb_per_year"])
    intercept = city_annual["mda8_o3"].median() - slope * city_annual["year"].median()
    fit = intercept + slope * city_annual["year"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(city_annual["year"], city_annual["mda8_o3"], marker="o", color="#1f6f8b", label="Annual mean")
    ax.plot(city_annual["year"], fit, linestyle="--", color="#9b2d20", label="Sen slope")
    ax.set_title("Citywide annual MDA8 O3 trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("MDA8 O3 (ppb)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "trend_citywide_annual_mda8_o3.png", dpi=220)
    plt.close(fig)


def plot_seasonal(city_seasonal: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for season in SEASON_ORDER:
        subset = city_seasonal[city_seasonal["season"] == season]
        ax.plot(
            subset["year"],
            subset["mda8_o3"],
            marker="o",
            linewidth=1.8,
            label=SEASON_LABELS[season],
        )
    ax.set_title("Citywide seasonal MDA8 O3 trends")
    ax.set_xlabel("Year")
    ax.set_ylabel("MDA8 O3 (ppb)")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "trend_citywide_seasonal_mda8_o3.png", dpi=220)
    plt.close(fig)


def plot_site_slopes(site_trend: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = site_trend.sort_values("sen_slope_ppb_per_year")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = np.where(plot_df["sen_slope_ppb_per_year"] < 0, "#1f6f8b", "#b45f06")
    ax.barh(plot_df["site"], plot_df["sen_slope_ppb_per_year"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Site-specific seasonal MK Sen slopes")
    ax.set_xlabel("Sen slope (ppb/year)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures_dir / "trend_by_site_sen_slopes.png", dpi=220)
    plt.close(fig)


def write_figures(trends: dict[str, pd.DataFrame], aggregates: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    plot_annual(aggregates["city_annual"], trends["trend_citywide_annual"], figures_dir)
    plot_seasonal(aggregates["city_seasonal"], figures_dir)
    plot_site_slopes(trends["trend_by_site_monthly_mk"], figures_dir)
    print(f"Wrote trend figures to {figures_dir}", flush=True)


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    daily = load_daily(Path(args.daily))
    aggregates = build_aggregates(daily)
    trends = run_trends(aggregates, args.alpha)
    write_tables(trends, aggregates, tables_dir)
    write_figures(trends, aggregates, figures_dir)

    city = trends["trend_citywide_monthly_mk"].iloc[0]
    print(
        "Citywide seasonal MK: "
        f"slope={city['sen_slope_ppb_per_year']:.3f} ppb/year, "
        f"tau={city['kendall_tau']:.3f}, p={city['p_value']:.4f}, trend={city['trend']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
