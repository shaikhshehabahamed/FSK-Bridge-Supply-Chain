"""
run_all.py — FSK Bridge Collapse Study: Full Reproducible Pipeline
==================================================================
Runs all 12 analytical steps in sequence. Figures and tables are written
directly to paper_assets/{figures,tables}/{main,supplementary}/ by each
step (main = 7 figures + 7 tables; supplementary = 6 figures + 6 tables) --
there is no separate copy/collation step.
Estimated runtime: 8-12 minutes (API rate limits dominate steps 1-3).

Usage:
  python run_all.py              # full pipeline
  python run_all.py --step 5    # run only step 5 onwards
  python run_all.py --data-only # collect data only (steps 1-3)
"""

import subprocess
import sys
import time
import os
from pathlib import Path

# ── Directory setup ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SCRIPTS = ROOT / "scripts"
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
PAPER_ASSETS = ROOT / "paper_assets"

for d in [DATA_RAW, DATA_PROC,
          PAPER_ASSETS / "figures" / "main", PAPER_ASSETS / "figures" / "supplementary",
          PAPER_ASSETS / "tables" / "main", PAPER_ASSETS / "tables" / "supplementary"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Pipeline steps ───────────────────────────────────────────────────────────
STEPS = [
    # (step_num, script_file, description)
    (1,  "01_fred_data.py",                      "FRED data collection (7 series, 2015-2024)"),
    (1,  "01b_fred_validation_plot.py",           "FRED validation: Figure S4"),
    (2,  "02_portwatch_data.py",                  "PortWatch RoRo daily data (4 ports)"),
    (2,  "02b_portwatch_container_metrics.py",    "PortWatch Baltimore monthly volume (complete)"),
    (2,  "02c_portwatch_validation_plot.py",      "PortWatch validation: Figure S5"),
    (3,  "03_comtrade_data.py",                   "UN Comtrade HS 8708 imports (2019-2024)"),
    (3,  "03b_comtrade_validation_plot.py",       "Comtrade validation: Figure S6"),
    (4,  "04_master_dataset.py",                  "Master panel assembly (72x35)"),
    (4,  "04b_eda.py",                            "EDA: descriptive stats + Figure 1"),
    (4,  "04c_eda_supplement.py",                 "EDA supplement: pre-trend + Figure S1"),
    (5,  "05_its_model.py",                       "ITS segmented regression (HAC-4)"),
    (5,  "05b_its_plots.py",                      "ITS plots: Figure 2 + Figure 3"),
    (6,  "06_safety_stock.py",                    "Implied safety stock inference"),
    (6,  "06b_safety_stock_plots.py",             "Safety stock: Figure 4"),
    (7,  "07_placebo_robustness.py",              "Placebo & robustness tests"),
    (7,  "07b_robustness_plots.py",               "Robustness: Figure S2"),
    (8,  "08_diversion_spillover.py",             "Port network diversion analysis"),
    (8,  "08b_diversion_plots.py",                "Diversion: Figure 5"),
    (8,  "08c_event_study.py",                   "Event-study specification"),
    (8,  "08d_event_study_plot.py",               "Event-study: Figure 6"),
    (9,  "09_welfare_estimate.py",                "Back-of-envelope welfare estimate"),
    (10, "10_arima_residuals.py",                 "ITS residual diagnostics: Figure S3"),
    (11, "11_carrying_cost.py",                   "Inventory carrying cost of the implied JIT buffer"),
    (11, "11b_carrying_cost_plot.py",             "Carrying cost: Figure 7"),
    (12, "09_tables.py",                          "Publication-ready result tables (7 main + 6 supplementary)"),
]

def run_script(script_name, description):
    script_path = SCRIPTS / script_name
    if not script_path.exists():
        print(f"  [SKIP] {script_name} not found")
        return True

    print(f"\n  Running: {script_name}")
    print(f"  {description}")
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(ROOT), env=env
    )

    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  Done in {elapsed:.1f}s")
        return True
    else:
        print(f"  FAILED in {elapsed:.1f}s")
        print(f"    STDERR: {result.stderr[-500:]}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1,
                        help="Start from this step number (default: 1)")
    parser.add_argument("--data-only", action="store_true",
                        help="Run only data collection (steps 1-3)")
    args = parser.parse_args()

    start_step = args.step
    end_step   = 3 if args.data_only else 99

    print("=" * 65)
    print("FSK BRIDGE COLLAPSE STUDY — Full Reproducible Pipeline")
    print("=" * 65)
    print(f"Starting from step {start_step}")
    print(f"Working directory: {ROOT}")

    total_t = time.time()
    passed, failed = 0, 0

    for step_num, script, desc in STEPS:
        if step_num < start_step or step_num > end_step:
            continue
        print(f"\n{'-'*65}")
        print(f"STEP {step_num}: {desc}")
        print(f"{'-'*65}")
        ok = run_script(script, desc)
        if ok:
            passed += 1
        else:
            failed += 1
            resp = input("\n  Script failed. Continue anyway? [y/N]: ")
            if resp.lower() != "y":
                print("  Aborting.")
                break

    print(f"\n{'='*65}")
    print(f"Pipeline complete: {passed} passed, {failed} failed")
    print(f"Total time: {(time.time()-total_t)/60:.1f} minutes")
    print(f"Figures: {PAPER_ASSETS / 'figures'}")
    print(f"Tables:  {PAPER_ASSETS / 'tables'}")

if __name__ == "__main__":
    main()
