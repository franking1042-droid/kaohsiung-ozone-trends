# Kaohsiung Surface Ozone Trends (2018–2025)

Analysis code, derived data tables, and submission figures for a study of surface
ozone (MDA8 O₃) trends in Kaohsiung, Taiwan, based on hourly observations from
12 air-quality monitoring stations (仁武 Renwu, 前金 Qianjin, 前鎮 Qianzhen,
大寮 Daliao, 小港 Xiaogang, 左營 Zuoying, 復興 Fuxing, 林園 Linyuan,
楠梓 Nanzi, 橋頭 Qiaotou, 美濃 Meinong, 鳳山 Fengshan).

The analysis combines:

- **Trend detection** — Seasonal Mann–Kendall test with Sen's slope on monthly
  mean MDA8 O₃.
- **Meteorological normalization** — Random-forest resampling normalization
  following Grange et al. (2018, *Atmos. Chem. Phys.*) and Grange & Carslaw
  (2019, *Sci. Total Environ.*), to separate meteorology-driven variability
  from the underlying emission-driven trend.
- **Driver models** — GAM and random-forest models of meteorological drivers of
  daily MDA8 O₃, with permutation importance.
- **High-ozone event analysis** — event definition (MDA8 > 60 ppb), seasonality,
  logistic-regression odds ratios, and station-type comparisons.

A manuscript based on this analysis is in preparation. The manuscript text is
not included in this repository.

## Repository layout

```
├── scripts/          # Analysis pipeline (01–13) + normalization module
├── data/             # Small derived data tables (see Data section)
├── figures/          # Submission figures (Fig 1–5, S1–S2)
├── tables/           # Figure caption list + Table S4 source data
└── requirements.txt
```

## Scripts

Scripts are numbered in pipeline order. Later "revision" scripts (10–13)
supersede parts of the earlier exploratory scripts (03–05, 09) and were the
ones used for the final manuscript numbers.

| Script | Purpose |
|---|---|
| `01_prepare_o3_meteo.py` | Parse raw hourly zip archives from the MOENV open-data platform into long-format hourly O₃/meteorology and daily MDA8 tables (QC: flag characters `#`, `*`, `x`, `A` and physically implausible values → NA; MDA8 requires ≥ 6 valid hours per 8-h window). |
| `02_descriptive_analysis.py` | Descriptive statistics and figures: annual, seasonal, diurnal, and spatial summaries. |
| `03_model_meteorological_drivers.py` | Exploratory GAM + random-forest models of meteorological drivers. |
| `04_high_o3_events.py` | High-ozone event identification and seasonality/meteorology composites. |
| `05_sensitivity_analysis.py` | Sensitivity analyses for the driver models. |
| `06_download_supplementary_data.py` | Download station metadata (MOENV AQX_P_07) and CWA CODiS daily meteorology/sunshine for station 467441. |
| `07_trend_analysis.py` | Seasonal Mann–Kendall trend tests and Sen's slope estimates. |
| `08_submission_robustness_checks.py` | Robustness checks prepared for submission. |
| `09_event60_classification.py` | Event (MDA8 > 60 ppb) classification models (logistic regression, random forest). |
| `10_revision_core_analyses.py` | Revision core analyses (T1: normalization main analysis and sensitivity). |
| `11_run_met_normalized_trend.py` | Wrapper that runs the meteorological-normalization module. |
| `_claude_met_normalized_trend.py` | The normalization module itself: RF resampling normalization + Seasonal Mann–Kendall / Sen's slope with confidence intervals. |
| `12_revision_t2_t3_models.py` | Revised GAM/RF driver models (T2/T3). |
| `13_revision_t4_t8_logistic_stationtype.py` | Logistic odds ratios and station-type comparison (T4/T8). |

Note: `11_run_met_normalized_trend.py` copies the normalization module from a
working-directory path that does not exist in this repository. The copied
module `scripts/_claude_met_normalized_trend.py` is included here — run it
directly, or point the wrapper's `SRC` path at it.

## Data

`data/` contains the small derived tables used by the revision scripts and the
manuscript's descriptive results:

| File | Content |
|---|---|
| `daily_mda8_o3_2018_2025.csv` | Station-day MDA8 O₃, 2018–2025 (core analysis table). |
| `annual_mda8_summary_by_site.csv` | Annual MDA8 summaries by station. |
| `seasonal_mda8_summary_by_site_year.csv` | Seasonal MDA8 summaries by station-year. |
| `diurnal_o3_profile_by_site_season.csv` | Mean diurnal O₃ profiles by station and season. |
| `station_mda8_spatial_summary.csv` | Spatial summary across stations. |
| `data_availability_by_site_year.csv` | Data availability by station-year. |
| `kaohsiung_air_quality_station_metadata.csv` | Air-quality station metadata (coordinates, district, station type). |
| `cwa_kaohsiung_station_metadata.csv` | CWA weather station metadata. |
| `cwa_kaohsiung_daily_meteorology_2025.csv` | CWA CODiS daily meteorology for 2025. |
| `cwa_air_station_daily_validation_2025.csv` | Cross-validation of CWA vs. air-station daily meteorology, 2025. |
| `kaohsiung_sunshine_daily_2016_2025_extended.csv` | Daily sunshine duration, 2016–2025. |

**Not included** (too large for GitHub; both are reproducible):

- Raw hourly zip archives (~300 MB) — download from the Taiwan Ministry of
  Environment open-data platform (https://data.moenv.gov.tw/ and the historical
  data section of https://airtw.moenv.gov.tw/).
- `hourly_o3_meteo_2018_2025.csv` (~90 MB) — regenerate from the raw archives
  with `scripts/01_prepare_o3_meteo.py`.

Meteorology and sunshine data come from the Central Weather Administration
CODiS platform (https://codis.cwa.gov.tw/), station 467441 (Kaohsiung).

## Figures

`figures/` holds the submission figures; `tables/圖表Caption清單.md` lists the
full captions.

- Fig 1 — Station map
- Fig 2 — Annual and monthly MDA8 O₃
- Fig 3 — Observed vs. meteorologically normalized trend
- Fig 4 — Event seasonality and meteorology
- Fig 5 — Random-forest variable importance
- Fig S1 — Diurnal profiles
- Fig S2 — Day-of-year cyclic-encoding comparison

## Setup

```
pip install -r requirements.txt
```

Python ≥ 3.10. Scripts write outputs to `outputs/` and `analysis_outputs/`
(git-ignored) relative to the repository root.
