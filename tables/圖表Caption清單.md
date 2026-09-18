# Figure and table captions (submission package)

All figure files are in `投稿打包/figures/` (300 dpi, Arial, 174 mm double-column width). Table files are in `投稿打包/tables/`.

## Main figures

**Fig. 1** Locations and official types of the 12 governmental air-quality monitoring stations in Kaohsiung, Taiwan. Station coordinates and type classifications (general, background, traffic, industrial) follow the Ministry of Environment monitoring-station metadata. File: `Fig1_station_map.png`.

**Fig. 2** Citywide mean MDA8 O3 in Kaohsiung, 2018-2025: (a) annual means; (b) monthly distributions of city-day values pooled across years. Citywide values are the mean of station-day MDA8 O3 across available stations on each day. Boxes show the median and interquartile range; whiskers extend to 1.5 times the interquartile range; dots are outliers. File: `Fig2_annual_monthly_mda8.png`.

**Fig. 3** Observed and meteorologically normalized citywide monthly mean MDA8 O3, 2018-2025. The normalized series is from the 300-iteration random-forest meteorological normalization. The inset reports Seasonal Mann-Kendall Sen slopes for the 2018-2025 window; the observed trend is not statistically significant, whereas the normalized trend is significantly negative. File: `Fig3_observed_vs_normalized_trend.png`.

**Fig. 4** Extreme O3 events at the city-day level, 2018-2025: (a) monthly counts of extreme O3 event days (city-day maximum MDA8 O3 >= the study-period 90th percentile, 80.34 ppb; 293 days in total); (b-d) month-adjusted anomalies of relative humidity, sunshine duration, and rainfall on event versus non-event days. Anomalies subtract the long-term calendar-month mean from each city-day value; the dashed line marks zero anomaly. Boxes show the median and interquartile range; whiskers extend to 1.5 times the interquartile range; outliers are omitted for clarity. Sample sizes differ slightly for sunshine because two days lacked sunshine data. File: `Fig4_event_seasonality_meteorology.png`.

**Fig. 5** Permutation importance of predictors in the revised city-day random-forest model of MDA8 O3 (held-out 2024-2025 test set; five repeats). Bars show mean importance and error bars show the standard deviation across repeats. File: `Fig5_rf_importance.png`.

## Supplementary figures

**Fig. S1** Seasonal diurnal profiles of citywide mean hourly O3 in Kaohsiung, 2018-2025. Lines show the mean hourly O3 concentration by season, averaged across the 12 monitoring stations. All seasons peak in the early afternoon (12:00-14:00 LST); the amplitude is largest in autumn and smallest in summer. File: `FigS1_diurnal_profiles.png`.

**Fig. S2** GAM partial effect of day of year on city-day MDA8 O3 under the original non-cyclic specification and the revised cyclic spline specification. The cyclic spline removes the artificial discontinuity between December 31 and January 1. File: `FigS2_doy_cyclic_comparison.png`.

## Main tables

**Table 1** Kaohsiung air-quality monitoring stations and MDA8 O3 data completeness, 2018-2025. (Station, type, coordinates, valid MDA8 station-days, minimum station-year availability.)

**Table 2** Seasonal Mann-Kendall trends of citywide monthly MDA8 O3 (observed and meteorologically normalized; three analysis windows) and station-level meteorologically normalized trends, 2018-2025.

**Table 3** High-O3 event prevalence by definition (P90 and P95 extreme events; 60 ppb exceedance days) and month-adjusted meteorological differences between event and non-event city-days (Mann-Whitney U tests with rank-biserial effect sizes).

**Table 4** Held-out (2024-2025) performance of revised and original meteorological driver models of MDA8 O3 (random forest and GAM; city-day main models and station-day sensitivity models).

## Supplementary tables

**Table S1** Month-adjusted logistic regression odds ratios (per 1 SD) for P90 and P95 extreme O3 events and 60 ppb exceedance days, with model summary statistics (McFadden pseudo-R2, in-sample ROC AUC, average precision, Brier score).

**Table S2** Station-level and station-type contrasts: MDA8 O3 summaries, mean NOx, city-maximum contribution counts, and meteorologically normalized Sen slopes, 2018-2025.

**Table S3** Month-adjusted meteorological anomalies on P95 extreme O3 event days versus non-event days, 2018-2025.

**Table S4** Seasonal Mann-Kendall trends of citywide monthly mean precursor and co-pollutant concentrations (NOx, NO2, CO, PM2.5), 2018-2025.

## Notes for assembly

- The 60 ppb classification-related material is placed in the supplementary tables (Table S1), per the submission decision.
- Fig. 1 currently uses a coordinate scatter without a GIS basemap; an optional upgrade is a basemap/GIS version before submission.
- Figure callouts in the Results and Discussion drafts match this numbering (Fig. 2-5, Fig. S1-S2; Tables 1-4, S1-S4). Fig. 1 is cited in the Methods study-area section.
- Supplementary tables file: `Tables_補充_TableS1-S4.docx`.
