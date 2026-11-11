"""
Step 2: IMF PortWatch Data Collection
Pulls daily port-level data for Baltimore (treatment) and
Port of Virginia + Brunswick (diversion ports) via ArcGIS Feature Service.
Aggregates to monthly. Saves raw + monthly CSVs.
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"

import requests
import pandas as pd
import json
import time
import warnings
warnings.filterwarnings("ignore")

BASE = ("https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/"
        "services/Daily_Ports_Data/FeatureServer/0/query")

PORTS = {
    "baltimore"  : "port103",
    "brunswick"  : "port180",
    "port_virginia": "port2066",
    "norfolk"    : "port826",
}

FIELDS = ("date,year,month,day,portid,portname,"
          "portcalls,portcalls_roro,"
          "import_roro,export_roro,"
          "import_cargo,export_cargo,"
          "import,export")

START_YEAR = 2019
END_YEAR   = 2024

def fetch_port(portid, portname_label):
    """Fetch all daily records for one port, 2019–2024."""
    all_records = []
    offset = 0
    page_size = 2000

    where = (f"portid='{portid}' AND year>={START_YEAR} AND year<={END_YEAR}")

    print(f"\n  Fetching {portname_label} ({portid}) ...")
    while True:
        params = {
            "where"          : where,
            "outFields"      : FIELDS,
            "resultOffset"   : offset,
            "resultRecordCount": page_size,
            "f"              : "json",
            "orderByFields"  : "date ASC",
        }
        resp = requests.get(BASE, params=params, timeout=60)
        data = resp.json()

        features = data.get("features", [])
        if not features:
            break

        for feat in features:
            all_records.append(feat["attributes"])

        print(f"    offset={offset}  got={len(features)}  total_so_far={len(all_records)}")

        if not data.get("exceededTransferLimit", False):
            break
        offset += page_size
        time.sleep(0.3)

    df = pd.DataFrame(all_records)
    if df.empty:
        print(f"    WARNING: No data returned for {portid}")
        return df

    # Parse date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"    Range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"    Records: {len(df)}")
    return df

# ── Fetch all ports ──────────────────────────────────────────────────────────
print("=" * 65)
print("PORTWATCH DATA PULL  (ArcGIS Feature Service)")
print("=" * 65)

raw_frames = {}
for label, pid in PORTS.items():
    df = fetch_port(pid, label)
    if not df.empty:
        df["port_label"] = label
        raw_frames[label] = df

# ── Save raw daily data ──────────────────────────────────────────────────────
raw_all = pd.concat(raw_frames.values(), ignore_index=True)
raw_all.to_csv(str(DATA_RAW / "portwatch_daily_raw.csv"), index=False)
print(f"\nRaw daily saved: {len(raw_all)} rows across {len(raw_frames)} ports")

# ── Aggregate to monthly ─────────────────────────────────────────────────────
raw_all["month_start"] = raw_all["date"].dt.to_period("M").dt.to_timestamp()

monthly = (raw_all
           .groupby(["port_label","month_start"])
           .agg(
               portcalls      =("portcalls",      "sum"),
               portcalls_roro =("portcalls_roro",  "sum"),
               import_roro    =("import_roro",     "sum"),
               export_roro    =("export_roro",     "sum"),
               import_cargo   =("import_cargo",    "sum"),
               export_cargo   =("export_cargo",    "sum"),
               import_total   =("import",          "sum"),
               export_total   =("export",          "sum"),
               n_days         =("date",            "count"),
           )
           .reset_index()
           .rename(columns={"month_start": "date"})
          )

# Refuses to overwrite the existing frozen file if a fresh pull disagrees with
# it beyond a wide tolerance -- same guard 02b_portwatch_container_metrics.py
# already uses for its own output, extended here to every numeric column
# (compared per date+port_label) since 06_safety_stock.py and
# 08_diversion_spillover.py both read portcalls_roro/import_roro shares from
# this file for the headline theta/diversion estimates. portwatch_monthly.csv
# is the frozen snapshot every number in the manuscript was verified against
# (see scripts/verify_claims.py); `python run_all.py --step 4` reproduces the
# full analysis from it without needing to re-collect.
target = DATA_RAW / "portwatch_monthly.csv"
NUMERIC_COLS = ["portcalls", "portcalls_roro", "import_roro", "export_roro",
                "import_cargo", "export_cargo", "import_total", "export_total"]

if target.exists():
    existing = pd.read_csv(target, parse_dates=["date"])
    common = existing.merge(monthly, on=["date", "port_label"], suffixes=("_old", "_new"))
    rel_diffs = [
        ((common[f"{c}_new"] - common[f"{c}_old"]).abs()
         / common[f"{c}_old"].abs().clip(lower=1)).median()
        for c in NUMERIC_COLS if f"{c}_old" in common.columns
    ]
    worst_rel_diff = max(rel_diffs) if rel_diffs and len(common) > 0 else 0.0
    if worst_rel_diff > 0.05 or abs(len(monthly) - len(existing)) > 5:
        print(f"\n  WARNING: fresh pull disagrees with the existing frozen "
              f"{target.name} (worst median relative diff={worst_rel_diff:.1%}, "
              f"row counts {len(monthly)} vs {len(existing)}).")
        print("  NOT overwriting the frozen file -- it is the snapshot every")
        print("  number in the manuscript was verified against. Fresh pull")
        print("  saved separately for inspection at "
              f"{target.with_name('portwatch_monthly_FRESH_PULL.csv').name}.")
        monthly.to_csv(target.with_name("portwatch_monthly_FRESH_PULL.csv"), index=False)
        print("Step 2 complete (frozen file preserved) ✓")
        raise SystemExit(0)
    else:
        print(f"\n  Fresh pull matches existing frozen file within tolerance "
              f"(worst median relative diff={worst_rel_diff:.2%}) -- safe to overwrite.")

monthly.to_csv(target, index=False)
print(f"Monthly agg saved: {len(monthly)} rows")

# ── Quick validation ─────────────────────────────────────────────────────────
print("\n── Baltimore RoRo port calls around FSK collapse ──")
balt = monthly[monthly["port_label"] == "baltimore"].copy()
balt = balt.set_index("date").loc["2024-01-01":"2024-09-01"]
print(balt[["portcalls_roro","import_roro","export_roro","n_days"]].to_string())

print("\n── Brunswick RoRo port calls (diversion signal) ──")
brun = monthly[monthly["port_label"] == "brunswick"].copy()
brun = brun.set_index("date").loc["2024-01-01":"2024-09-01"]
print(brun[["portcalls_roro","import_roro","export_roro","n_days"]].to_string())

print("\nStep 2 — PortWatch pull complete ✓")
