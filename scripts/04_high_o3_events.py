from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EVENT_FEATURES = [
    "mda8_o3",
    "temp_max",
    "rh_mean",
    "wind_speed_mean",
    "rain_sum",
    "sunshine_hours",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify high-MDA8 O3 event days and summarize meteorology.")
    parser.add_argument("--daily", default="outputs/tables/daily_mda8_o3_2018_2025.csv")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--quantile", type=float, default=0.90, help="Citywide MDA8 max quantile used as event cutoff.")
    return parser.parse_args()


def wind_sector(degrees: float | int | None) -> str:
    if pd.isna(degrees):
        return "missing"
    sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((float(degrees) + 22.5) % 360) // 45)
    return sectors[idx]


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(args.daily, parse_dates=["date"])
    city = (
        daily.groupby("date", as_index=False)
        .agg(
            city_mean_mda8=("mda8_o3", "mean"),
            city_max_mda8=("mda8_o3", "max"),
            affected_sites=("mda8_o3", lambda x: x.notna().sum()),
            temp_max=("temp_max", "mean"),
            rh_mean=("rh_mean", "mean"),
            wind_speed_mean=("wind_speed_mean", "mean"),
            rain_sum=("rain_sum", "mean"),
            sunshine_hours=("sunshine_hours", "mean"),
        )
        .dropna(subset=["city_max_mda8"])
    )
    threshold = city["city_max_mda8"].quantile(args.quantile)
    event_dates = set(city.loc[city["city_max_mda8"] >= threshold, "date"])
    city["high_o3_event"] = city["date"].isin(event_dates)
    city["event_threshold_ppb"] = threshold

    site_threshold = daily["mda8_o3"].quantile(args.quantile)
    daily["high_o3_event_day"] = daily["date"].isin(event_dates)
    daily["site_high_mda8"] = daily["mda8_o3"] >= site_threshold
    daily["wind_sector"] = daily["wind_dir_mean"].map(wind_sector)

    event_table = city[city["high_o3_event"]].sort_values("city_max_mda8", ascending=False)
    event_table.to_csv(tables_dir / "high_o3_event_days.csv", index=False, encoding="utf-8-sig")
    event_table.assign(month=event_table["date"].dt.month).groupby("month", as_index=False).size().rename(
        columns={"size": "n_event_days"}
    ).to_csv(tables_dir / "high_o3_event_counts_by_month.csv", index=False, encoding="utf-8-sig")
    event_table.assign(season=event_table["date"].dt.month.map(month_to_season)).groupby(
        "season", as_index=False
    ).size().rename(columns={"size": "n_event_days"}).to_csv(
        tables_dir / "high_o3_event_counts_by_season.csv", index=False, encoding="utf-8-sig"
    )

    available_features = [feature for feature in EVENT_FEATURES if feature in daily.columns]
    summary_rows = []
    for feature in available_features:
        grouped = daily.groupby("high_o3_event_day")[feature]
        for is_event, values in grouped:
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
        tables_dir / "high_o3_event_meteorology_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    daily["month"] = daily["date"].dt.month
    anomaly_rows = []
    for feature in available_features:
        climatology = daily.groupby(["site", "month"])[feature].transform("mean")
        anomaly = daily[feature] - climatology
        tmp = pd.DataFrame({"high_o3_event_day": daily["high_o3_event_day"], "anomaly": anomaly})
        for is_event, values in tmp.groupby("high_o3_event_day")["anomaly"]:
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
        tables_dir / "high_o3_event_meteorology_month_adjusted_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sector = (
        daily.groupby(["high_o3_event_day", "wind_sector"], as_index=False)
        .size()
        .rename(columns={"size": "n_site_days"})
    )
    sector["period"] = sector["high_o3_event_day"].map({True: "event", False: "non_event"})
    sector.to_csv(tables_dir / "high_o3_event_wind_sector_counts.csv", index=False, encoding="utf-8-sig")

    plot_df = daily.melt(
        id_vars=["high_o3_event_day"],
        value_vars=[feature for feature in ["temp_max", "rh_mean", "wind_speed_mean", "rain_sum"] if feature in daily.columns],
        var_name="feature",
        value_name="value",
    ).dropna()
    plot_df["period"] = plot_df["high_o3_event_day"].map({True: "event", False: "non-event"})

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), squeeze=False)
    for ax, feature in zip(axes.ravel(), plot_df["feature"].unique()):
        subset = plot_df[plot_df["feature"] == feature]
        sns.boxplot(data=subset, x="period", y="value", ax=ax, color="#b7d6cc", fliersize=1.5)
        ax.set_title(feature)
        ax.set_xlabel("")
    for ax in axes.ravel()[plot_df["feature"].nunique() :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "high_o3_event_meteorology_boxplots.png", dpi=220)
    plt.close(fig)

    print(
        f"Event cutoff: citywide daily max MDA8 >= {threshold:.2f} ppb "
        f"(quantile={args.quantile}). Wrote {len(event_table)} event days.",
        flush=True,
    )


def month_to_season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


if __name__ == "__main__":
    main()
