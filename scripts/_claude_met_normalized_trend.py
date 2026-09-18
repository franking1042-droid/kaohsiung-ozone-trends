#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
met_normalized_trend.py
=======================================================================
高雄 MDA8 O3 氣象正規化趨勢分析 (meteorological normalization)

方法:Grange et al. (2018, Atmos. Chem. Phys.) / Grange & Carslaw
(2019, Sci. Total Environ.) 的 Random Forest 重抽樣正規化,
配合 Seasonal Mann-Kendall 檢定與 Sen's slope(含信賴區間)。

【與論文 2.8 節「預測型」RF 的差異 —— 這段請寫進論文方法】
  1. 本模型以 2018-2025 全期資料訓練。目的是解釋 / 正規化,
     不是對未來外推預測,因此不做 chronological split;
     趨勢項 date_num 在此脈絡是合法且必要的。
  2. 正規化程序:氣象變數 (met_cols) 自「同測站、全研究期間」的
     觀測日 *整列聯合重抽樣*(joint resampling,保留氣象變數間的
     協方差結構,避免產生高溫+高濕+強降雨這類不符物理的組合);
     趨勢項 date_num、季節項 doy、測站身分保持原值不動。
  3. 因 doy 保持原值,正規化後的序列仍保留季節循環,
     故後續仍以「月分層 Seasonal Mann-Kendall」檢定,
     與論文原本對觀測值的趨勢分析可以直接並列對照。
  4. 另提供 deseasonalized 變體(連 doy 一起重抽)供趨勢視覺化。

【執行前唯一必要動作】修改下方 CONFIG 的檔案路徑與欄位名稱。

輸出(皆存至 CONFIG["output_dir"]):
  normalized_station_day.csv          站日層級:觀測值 vs 正規化值
  monthly_observed_vs_normalized.csv  全市月均序列
  trend_results.csv                   各時間窗 x (觀測/正規化) 趨勢結果
  trend_results_by_station.csv        逐站趨勢(可選)
  fig_met_normalized_trend.png        對照圖
  run_log.txt                         執行紀錄(含 OOB R2、刪除筆數)

相依套件:pandas, numpy, scikit-learn, matplotlib
(pymannkendall 為可選,若安裝會另做交叉驗證)
=======================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

# =====================================================================
# CONFIG —— Codex / 使用者:只需要改這一區
# =====================================================================
CONFIG = {
    # ---- 輸入輸出 ----
    "input_csv": "station_day_mda8.csv",   # ← 改成你的站日表路徑
    "output_dir": "met_norm_outputs",

    # ---- 欄位對應(左邊固定,右邊改成你資料的實際欄名)----
    "col_date":   "date",        # 日期欄(可被 pd.to_datetime 解析)
    "col_site":   "site",        # 測站名稱/代碼欄
    "col_target": "mda8_o3",     # MDA8 O3 (ppb)

    # 要被重抽樣的氣象欄位 = 論文 2.8 節主模型的氣象預測變數。
    # 若欄名不同請逐一修改;若某欄不存在,腳本會自動略過並記錄。
    "met_cols": ["t_max", "rh_mean", "ws_mean",
                 "rain_log1p", "wd_sin", "wd_cos"],

    # 若資料只有原始日雨量、還沒有 log1p 欄,填原始雨量欄名,
    # 腳本會自動建立 rain_log1p;若已有 log1p 欄則設為 None。
    "col_rain_raw": None,

    # 日照時數(城市層級單一測站,論文的第二組模型)。
    # 預設 False = 主模型不含日照,與 2.8 節主規格一致。
    "include_sunshine": False,
    "col_sunshine": "sunshine_hr",

    # ---- 模型與正規化參數 ----
    "n_trees": 500,
    "min_leaf": 5,
    "n_iterations": 300,     # 重抽樣次數;快速測試可降到 100
    "random_seed": 42,

    # False = 經典 Grange 作法:donor 日取自同測站全期間任意日
    #         (季節性完全由 doy 項吸收)。
    # True  = donor 限制在同測站、同「月份」(跨年份)之內,
    #         只移除該月內的氣象異常、保留各月氣象氣候值。
    #         兩者皆可辯護;預設 False,可各跑一次做敏感度。
    "resample_within_month": False,

    # ---- 趨勢分析 ----
    # 與論文 2.6 節的三個 robustness 時間窗一致
    "windows": [
        ("2018-01-01", "2025-12-31"),
        ("2018-01-01", "2024-12-31"),
        ("2019-01-01", "2025-12-31"),
    ],
    "min_days_per_month": 20,   # 月均值有效性門檻
    "alpha": 0.05,

    # ---- 其他 ----
    "run_station_level": True,      # 逐站正規化趨勢
    "deseasonalized_variant": True, # 額外跑「連 doy 一起重抽」版本
}

LOG_LINES = []


def log(msg=""):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


# =====================================================================
# 1. 資料載入與特徵準備
# =====================================================================
def load_and_prepare(cfg):
    df = pd.read_csv(cfg["input_csv"])
    log(f"[load] 讀入 {len(df):,} 列,欄位:{list(df.columns)}")

    df[cfg["col_date"]] = pd.to_datetime(df[cfg["col_date"]])
    df = df.sort_values([cfg["col_site"], cfg["col_date"]]).reset_index(drop=True)

    # 若需要,建立 rain_log1p
    if cfg["col_rain_raw"]:
        df["rain_log1p"] = np.log1p(
            pd.to_numeric(df[cfg["col_rain_raw"]], errors="coerce").clip(lower=0)
        )
        log(f"[prep] 由 {cfg['col_rain_raw']} 建立 rain_log1p")

    # 衍生時間變數
    d = df[cfg["col_date"]]
    df["_date_num"] = (d - d.min()).dt.days.astype(float)  # 趨勢項
    df["_doy"] = d.dt.dayofyear.astype(float)              # 季節項
    df["_year"] = d.dt.year
    df["_month"] = d.dt.month

    # 確認氣象欄位存在
    met_cols = [c for c in cfg["met_cols"] if c in df.columns]
    missing = [c for c in cfg["met_cols"] if c not in df.columns]
    if missing:
        log(f"[warn] 下列氣象欄位不存在、將略過:{missing}")
    if cfg["include_sunshine"]:
        if cfg["col_sunshine"] in df.columns:
            met_cols = met_cols + [cfg["col_sunshine"]]
            log(f"[prep] 已納入日照欄位 {cfg['col_sunshine']}")
        else:
            log(f"[warn] include_sunshine=True 但找不到欄位 {cfg['col_sunshine']}")
    if not met_cols:
        raise ValueError("沒有任何可用的氣象欄位,請檢查 CONFIG['met_cols']。")

    # 建模用列:target 與所有特徵皆非缺值
    need = [cfg["col_target"], "_date_num", "_doy"] + met_cols
    n0 = len(df)
    model_mask = df[need].notna().all(axis=1)
    n_drop = int((~model_mask).sum())
    log(f"[prep] 特徵/目標含缺值而排除建模的列數:{n_drop:,} / {n0:,} "
        f"({100 * n_drop / n0:.2f}%)")

    return df, met_cols, model_mask


def build_design(df, met_cols, cfg):
    """回傳特徵矩陣 (ndarray)、特徵名稱清單、測站 one-hot 欄名。"""
    site_dummies = pd.get_dummies(df[cfg["col_site"]], prefix="site")
    feature_df = pd.concat(
        [df[["_date_num", "_doy"] + met_cols], site_dummies.astype(float)],
        axis=1,
    )
    feature_cols = list(feature_df.columns)
    return feature_df.to_numpy(dtype=float), feature_cols


# =====================================================================
# 2. RF 訓練與氣象正規化
# =====================================================================
def train_rf(Xmat, y, cfg):
    rf = RandomForestRegressor(
        n_estimators=cfg["n_trees"],
        min_samples_leaf=cfg["min_leaf"],
        oob_score=True,
        n_jobs=-1,
        random_state=cfg["random_seed"],
    )
    rf.fit(Xmat, y)
    log(f"[rf] 訓練完成:n={len(y):,}, trees={cfg['n_trees']}, "
        f"min_leaf={cfg['min_leaf']}")
    log(f"[rf] OOB R2 = {rf.oob_score_:.3f}(全期資料;解釋/正規化用途)")
    return rf


def _donor_groups(df, mask, cfg):
    """建立 donor 抽樣分組:同測站(可選:同月份)。回傳 {key: 索引陣列}。"""
    sub = df.loc[mask]
    if cfg["resample_within_month"]:
        keys = list(zip(sub[cfg["col_site"]].values, sub["_month"].values))
    else:
        keys = list(sub[cfg["col_site"]].values)
    groups = {}
    for pos, k in zip(np.flatnonzero(mask.to_numpy()), keys):
        groups.setdefault(k, []).append(pos)
    return {k: np.asarray(v) for k, v in groups.items()}


def normalize(rf, Xmat, feature_cols, groups, met_cols, cfg,
              also_resample_doy=False, label="met"):
    """
    重抽樣正規化主迴圈。
    對每次疊代:每個 donor 分組內,隨機(取後放回)抽 donor 列,
    以其氣象欄位(joint,整列一起)覆蓋原列;預測後累加,最後取平均。
    """
    rng = np.random.default_rng(cfg["random_seed"] + (1 if also_resample_doy else 0))
    swap_cols = list(met_cols) + (["_doy"] if also_resample_doy else [])
    swap_idx = np.array([feature_cols.index(c) for c in swap_cols])

    pred_sum = np.zeros(Xmat.shape[0])
    n_iter = cfg["n_iterations"]
    for it in range(n_iter):
        Xs = Xmat.copy()
        for _, idx in groups.items():
            donor = rng.choice(idx, size=idx.size, replace=True)
            Xs[np.ix_(idx, swap_idx)] = Xmat[np.ix_(donor, swap_idx)]
        pred_sum += rf.predict(Xs)
        if (it + 1) % 25 == 0 or it == 0:
            log(f"[normalize:{label}] iteration {it + 1}/{n_iter}")
    return pred_sum / n_iter


# =====================================================================
# 3. Seasonal Mann-Kendall + Sen's slope(手刻,單位明確為 ppb/year)
# =====================================================================
def seasonal_mk_sen(monthly, value_col, alpha=0.05):
    """
    monthly:含 _year, _month, value_col 的 DataFrame(每月至多一筆)。
    月為季節層;S、Var(S) 逐月累加(含 ties 修正);
    Sen slope = 同月跨年配對斜率的中位數(分母為年差 → 單位 ppb/year);
    CI 依 Gilbert (1987) 常態近似(ties 於 CI 中忽略,屬近似)。
    """
    S, VarS = 0.0, 0.0
    slopes = []
    n_used = 0
    for m in range(1, 13):
        sub = monthly.loc[monthly["_month"] == m, ["_year", value_col]].dropna()
        sub = sub.sort_values("_year")
        y = sub[value_col].to_numpy(float)
        t = sub["_year"].to_numpy(float)
        n = len(y)
        n_used += n
        if n < 2:
            continue
        for i in range(n - 1):
            dy = y[i + 1:] - y[i]
            dt = t[i + 1:] - t[i]
            S += np.sign(dy).sum()
            slopes.extend((dy / dt).tolist())
        vals, counts = np.unique(y, return_counts=True)
        tie = float((counts * (counts - 1) * (2 * counts + 5)).sum())
        VarS += (n * (n - 1) * (2 * n + 5) - tie) / 18.0

    slopes = np.sort(np.asarray(slopes, dtype=float))
    Np = slopes.size
    if Np == 0 or VarS <= 0:
        return dict(n_months=n_used, S=S, VarS=VarS, z=np.nan, p=np.nan,
                    slope=np.nan, ci_low=np.nan, ci_high=np.nan)

    if S > 0:
        z = (S - 1) / np.sqrt(VarS)
    elif S < 0:
        z = (S + 1) / np.sqrt(VarS)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    slope = float(np.median(slopes))
    Calpha = stats.norm.ppf(1 - alpha / 2) * np.sqrt(VarS)
    M1 = int(np.floor((Np - Calpha) / 2))
    M2 = int(np.ceil((Np + Calpha) / 2))
    ci_low = float(slopes[max(M1, 0)])
    ci_high = float(slopes[min(M2, Np - 1)])
    return dict(n_months=n_used, S=float(S), VarS=float(VarS), z=float(z),
                p=float(p), slope=slope, ci_low=ci_low, ci_high=ci_high)


def optional_pymannkendall_check(series_monthly):
    """若環境裝有 pymannkendall,以其 seasonal_test 交叉驗證 p 值方向。"""
    try:
        import pymannkendall as mk
    except ImportError:
        log("[mk-check] 未安裝 pymannkendall,略過交叉驗證(非必要)。")
        return
    s = series_monthly.dropna().to_numpy()
    if s.size >= 24:
        res = mk.seasonal_test(s, period=12)
        log(f"[mk-check] pymannkendall.seasonal_test:trend={res.trend}, "
            f"p={res.p:.4f}(僅供交叉比對;正式數字採手刻版含明確單位)")


# =====================================================================
# 4. 聚合、時間窗、逐站分析、繪圖
# =====================================================================
def to_city_monthly(df, cfg, value_cols):
    daily = (df.groupby(cfg["col_date"])[value_cols]
               .mean()
               .rename_axis("date"))
    counts = df.groupby(cfg["col_date"])[value_cols[0]].size()
    monthly = daily.resample("MS").mean()
    day_counts = counts.resample("MS").size()
    monthly.loc[day_counts < cfg["min_days_per_month"]] = np.nan
    monthly["_year"] = monthly.index.year
    monthly["_month"] = monthly.index.month
    return daily, monthly


def run_windows(monthly, cfg, series_map):
    rows = []
    for (w0, w1) in cfg["windows"]:
        sub = monthly.loc[w0:w1]
        for label, col in series_map.items():
            r = seasonal_mk_sen(sub, col, alpha=cfg["alpha"])
            direction = ("increasing" if (r["p"] < cfg["alpha"] and r["slope"] > 0)
                         else "decreasing" if (r["p"] < cfg["alpha"] and r["slope"] < 0)
                         else "no significant trend")
            rows.append({
                "series": label, "window": f"{w0[:4]}-{w1[:4]}",
                "n_months": r["n_months"],
                "sen_slope_ppb_per_yr": r["slope"],
                "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                "z": r["z"], "p_value": r["p"], "direction": direction,
            })
    return pd.DataFrame(rows)


def run_station_level(df, cfg):
    rows = []
    for site, g in df.groupby(cfg["col_site"]):
        m = (g.set_index(cfg["col_date"])[["o3_norm", cfg["col_target"]]]
               .resample("MS").mean())
        m["_year"], m["_month"] = m.index.year, m.index.month
        for label, col in [("observed", cfg["col_target"]),
                           ("met_normalized", "o3_norm")]:
            r = seasonal_mk_sen(m, col, alpha=cfg["alpha"])
            rows.append({"site": site, "series": label,
                         "sen_slope_ppb_per_yr": r["slope"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                         "p_value": r["p"]})
    return pd.DataFrame(rows)


def make_figure(monthly, daily, trend_df, cfg, out_png, has_deseason):
    n_panel = 2 if has_deseason else 1
    fig, axes = plt.subplots(n_panel, 1, figsize=(11, 4.2 * n_panel),
                             sharex=True)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.plot(monthly.index, monthly[cfg["col_target"]],
            color="0.55", lw=1.2, label="Observed monthly MDA8 O3")
    ax.plot(monthly.index, monthly["o3_norm"],
            color="#c0392b", lw=1.8, label="Met-normalized monthly MDA8 O3")
    full = trend_df[(trend_df["window"] == "2018-2025")]
    txt = []
    for _, r in full.iterrows():
        txt.append(f"{r['series']}: Sen = {r['sen_slope_ppb_per_yr']:+.2f} "
                   f"ppb/yr [{r['ci_low']:.2f}, {r['ci_high']:.2f}], "
                   f"p = {r['p_value']:.3f}")
    ax.text(0.01, 0.02, "\n".join(txt), transform=ax.transAxes,
            fontsize=8.5, va="bottom",
            bbox=dict(fc="white", alpha=0.8, ec="0.7"))
    ax.set_ylabel("MDA8 O3 (ppb)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Citywide monthly MDA8 O3: observed vs meteorologically normalized")

    if has_deseason:
        ax2 = axes[1]
        ax2.plot(daily.index, daily["o3_norm_deseason"],
                 color="0.8", lw=0.5,
                 label="Deseasonalized normalized (daily)")
        roll = daily["o3_norm_deseason"].rolling(90, center=True,
                                                 min_periods=45).mean()
        ax2.plot(daily.index, roll, color="#2c3e50", lw=2,
                 label="90-day rolling mean")
        ax2.set_ylabel("MDA8 O3 (ppb)")
        ax2.legend(loc="upper right", fontsize=8.5)
        ax2.set_title("Deseasonalized meteorologically normalized series "
                      "(trend visualization)")

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    log(f"[fig] 已輸出 {out_png}")


# =====================================================================
# 5. 主流程
# =====================================================================
def main(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    log("=" * 68)
    log("Met-normalized trend analysis — MDA8 O3, Kaohsiung 2018-2025")
    log("=" * 68)

    df, met_cols, model_mask = load_and_prepare(cfg)
    dfm = df.loc[model_mask].reset_index(drop=True)

    Xmat, feature_cols = build_design(dfm, met_cols, cfg)
    y = dfm[cfg["col_target"]].to_numpy(float)
    rf = train_rf(Xmat, y, cfg)

    imp = (pd.Series(rf.feature_importances_, index=feature_cols)
             .sort_values(ascending=False).head(10))
    log("[rf] Impurity importance(前 10,僅供健全性檢查):")
    for k, v in imp.items():
        log(f"       {k:<16s} {v:.3f}")

    groups = _donor_groups(dfm.assign(_pos=np.arange(len(dfm))),
                           pd.Series(True, index=dfm.index), cfg)
    log(f"[normalize] donor 分組數:{len(groups)}"
        f"(within_month={cfg['resample_within_month']})")

    dfm["o3_norm"] = normalize(rf, Xmat, feature_cols, groups, met_cols,
                               cfg, also_resample_doy=False, label="met")

    has_deseason = bool(cfg["deseasonalized_variant"])
    if has_deseason:
        dfm["o3_norm_deseason"] = normalize(
            rf, Xmat, feature_cols, groups, met_cols, cfg,
            also_resample_doy=True, label="met+doy")

    # ---- 輸出站日層級 ----
    keep = [cfg["col_date"], cfg["col_site"], cfg["col_target"], "o3_norm"]
    if has_deseason:
        keep.append("o3_norm_deseason")
    out1 = os.path.join(cfg["output_dir"], "normalized_station_day.csv")
    dfm[keep].to_csv(out1, index=False)
    log(f"[out] {out1}")

    # ---- 全市月均與趨勢 ----
    value_cols = [cfg["col_target"], "o3_norm"] + (
        ["o3_norm_deseason"] if has_deseason else [])
    daily, monthly = to_city_monthly(dfm, cfg, value_cols)
    out2 = os.path.join(cfg["output_dir"],
                        "monthly_observed_vs_normalized.csv")
    monthly.to_csv(out2)
    log(f"[out] {out2}")

    trend_df = run_windows(monthly, cfg, {
        "observed": cfg["col_target"],
        "met_normalized": "o3_norm",
    })
    out3 = os.path.join(cfg["output_dir"], "trend_results.csv")
    trend_df.to_csv(out3, index=False)
    log(f"[out] {out3}")
    log("\n[結果摘要] Seasonal Mann-Kendall + Sen's slope(ppb/year):")
    log(trend_df.to_string(index=False,
                           float_format=lambda v: f"{v:.3f}"))

    optional_pymannkendall_check(monthly["o3_norm"])

    # ---- 逐站 ----
    if cfg["run_station_level"]:
        st = run_station_level(dfm, cfg)
        out4 = os.path.join(cfg["output_dir"],
                            "trend_results_by_station.csv")
        st.to_csv(out4, index=False)
        log(f"[out] {out4}")

    # ---- 圖 ----
    out_png = os.path.join(cfg["output_dir"],
                           "fig_met_normalized_trend.png")
    make_figure(monthly, daily, trend_df, cfg, out_png, has_deseason)

    # ---- log ----
    out_log = os.path.join(cfg["output_dir"], "run_log.txt")
    with open(out_log, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    log(f"[out] {out_log}")
    log("\n完成。論文引用寫法建議:Grange et al. (2018, ACP); "
        "Grange & Carslaw (2019, STOTEN)。")
    return dfm, monthly, trend_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        CONFIG["input_csv"] = sys.argv[1]
    main(CONFIG)
