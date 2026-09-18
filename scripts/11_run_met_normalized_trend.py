from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "claude建議" / "met_normalized_trend.py"
COPIED = ROOT / "scripts" / "_claude_met_normalized_trend.py"
OUT_ROOT = ROOT / "analysis_outputs" / "revision"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_module():
    if not COPIED.exists() or COPIED.stat().st_mtime < SRC.stat().st_mtime:
        shutil.copy2(SRC, COPIED)
    spec = importlib.util.spec_from_file_location("claude_met_normalized_trend", COPIED)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {COPIED}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one(module, name: str, *, n_iterations: int, resample_within_month: bool, include_sunshine: bool) -> None:
    module.LOG_LINES.clear()
    cfg = dict(module.CONFIG)
    cfg.update(
        {
            "input_csv": str(ROOT / "outputs" / "tables" / "daily_mda8_o3_2018_2025.csv"),
            "output_dir": str(OUT_ROOT / name),
            "col_date": "date",
            "col_site": "site",
            "col_target": "mda8_o3",
            "met_cols": [
                "temp_max",
                "rh_mean",
                "wind_speed_mean",
                "rain_log1p",
                "wind_dir_sin",
                "wind_dir_cos",
            ],
            "col_rain_raw": "rain_sum",
            "include_sunshine": include_sunshine,
            "col_sunshine": "sunshine_hours",
            "n_trees": 500,
            "min_leaf": 5,
            "n_iterations": n_iterations,
            "random_seed": 42,
            "resample_within_month": resample_within_month,
            "run_station_level": True,
            "deseasonalized_variant": True,
        }
    )
    print(
        f"[T1] Running {name}: iterations={n_iterations}, within_month={resample_within_month}, "
        f"sunshine={include_sunshine}",
        flush=True,
    )
    module.main(cfg)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    module = load_module()

    # Fast sanity check first, then formal runs.
    run_one(module, "T1_met_norm_quick100", n_iterations=100, resample_within_month=False, include_sunshine=False)
    run_one(module, "T1_met_norm_main300", n_iterations=300, resample_within_month=False, include_sunshine=False)
    run_one(module, "T1_met_norm_within_month300", n_iterations=300, resample_within_month=True, include_sunshine=False)
    run_one(module, "T1_met_norm_sunshine300", n_iterations=300, resample_within_month=False, include_sunshine=True)
    print(f"[T1] All outputs written under {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
