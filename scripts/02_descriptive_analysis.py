from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SEASON_ORDER = ["spring", "summer", "autumn", "winter"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create descriptive O3 trend, seasonal, and diurnal outputs.")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--figures-dir", default="outputs/figures")
    return parser.parse_args()


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Noto Sans CJK TC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_data(tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(tables_dir / "daily_mda8_o3_2018_2025.csv", parse_dates=["date"])
    hourly = pd.read_csv(tables_dir / "hourly_o3_meteo_2018_2025.csv", parse_dates=["datetime", "date"])
    return daily, hourly


def write_summary_tables(daily: pd.DataFrame, hourly: pd.DataFrame, tables_dir: Path) -> None:
    annual = (
        daily.groupby(["site", "year"], as_index=False)
        .agg(
            n_days=("mda8_o3", "count"),
            mean_mda8=("mda8_o3", "mean"),
            median_mda8=("mda8_o3", "median"),
            p90_mda8=("mda8_o3", lambda x: x.quantile(0.90)),
            p95_mda8=("mda8_o3", lambda x: x.quantile(0.95)),
            max_mda8=("mda8_o3", "max"),
        )
    )
    annual.to_csv(tables_dir / "annual_mda8_summary_by_site.csv", index=False, encoding="utf-8-sig")

    seasonal = (
        daily.groupby(["site", "year", "season"], as_index=False)
        .agg(
            n_days=("mda8_o3", "count"),
            mean_mda8=("mda8_o3", "mean"),
            p90_mda8=("mda8_o3", lambda x: x.quantile(0.90)),
        )
    )
    seasonal.to_csv(tables_dir / "seasonal_mda8_summary_by_site_year.csv", index=False, encoding="utf-8-sig")

    diurnal = (
        hourly.groupby(["site", "season", "hour"], as_index=False)
        .agg(mean_o3=("O3", "mean"), p90_o3=("O3", lambda x: x.quantile(0.90)), n_hours=("O3", "count"))
    )
    diurnal.to_csv(tables_dir / "diurnal_o3_profile_by_site_season.csv", index=False, encoding="utf-8-sig")


def plot_annual_city(daily: pd.DataFrame, figures_dir: Path) -> None:
    city_daily = daily.groupby(["date", "year"], as_index=False).agg(city_mean_mda8=("mda8_o3", "mean"))
    annual = city_daily.groupby("year", as_index=False).agg(mean_mda8=("city_mean_mda8", "mean"))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.lineplot(data=annual, x="year", y="mean_mda8", marker="o", ax=ax, color="#1f6f8b")
    ax.set_title("Kaohsiung citywide annual mean MDA8 O3")
    ax.set_xlabel("Year")
    ax.set_ylabel("MDA8 O3 (ppb)")
    ax.set_xticks(sorted(annual["year"].unique()))
    fig.tight_layout()
    fig.savefig(figures_dir / "annual_citywide_mda8_o3.png", dpi=220)
    plt.close(fig)


def plot_monthly_box(daily: pd.DataFrame, figures_dir: Path) -> None:
    city_daily = (
        daily.groupby(["date", "year", "month"], as_index=False)
        .agg(city_mean_mda8=("mda8_o3", "mean"))
        .dropna(subset=["city_mean_mda8"])
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    sns.boxplot(data=city_daily, x="month", y="city_mean_mda8", ax=ax, color="#b7d6cc", fliersize=2)
    ax.set_title("Monthly distribution of citywide MDA8 O3")
    ax.set_xlabel("Month")
    ax.set_ylabel("MDA8 O3 (ppb)")
    fig.tight_layout()
    fig.savefig(figures_dir / "monthly_citywide_mda8_o3_boxplot.png", dpi=220)
    plt.close(fig)


def plot_diurnal(hourly: pd.DataFrame, figures_dir: Path) -> None:
    profile = (
        hourly.groupby(["season", "hour"], as_index=False)
        .agg(mean_o3=("O3", "mean"))
        .dropna(subset=["mean_o3"])
    )
    profile["season"] = pd.Categorical(profile["season"], categories=SEASON_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    sns.lineplot(data=profile, x="hour", y="mean_o3", hue="season", marker="o", ax=ax)
    ax.set_title("Citywide diurnal O3 profile by season")
    ax.set_xlabel("Hour")
    ax.set_ylabel("O3 (ppb)")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(figures_dir / "diurnal_o3_by_season.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    daily, hourly = load_data(tables_dir)
    write_summary_tables(daily, hourly, tables_dir)
    plot_annual_city(daily, figures_dir)
    plot_monthly_box(daily, figures_dir)
    plot_diurnal(hourly, figures_dir)
    print(f"Wrote descriptive tables to {tables_dir}", flush=True)
    print(f"Wrote figures to {figures_dir}", flush=True)


if __name__ == "__main__":
    main()
