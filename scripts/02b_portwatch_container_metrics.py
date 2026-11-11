"""
Step 2b: Fetch Container_Metrics from PortWatch (POST-based)
This service has COMPLETE monthly coverage 2019-2024 — fills the gap in Daily_Ports_Data.
port103 = Baltimore, port180 = Brunswick, port2066 = Port of Virginia
Units: estimated monthly import volume in DWT-tons equivalent

REPRODUCIBILITY NOTE (added on re-verification): the live service now returns
FIVE rows per port per date (one per "metric": import_container,
export_container, incoming_cargo_container, outgoing_cargo_container,
portcalls) where originally it returned one. It also currently 400s on any
multi-field query that includes port180 (Brunswick) alongside another port
field. Both appear to be upstream API changes since this pipeline's original
data collection. This script (a) queries one port field at a time so a
problem with one port does not block the others, since only Baltimore is
required downstream, and (b) filters to metric='import_container' as the
closest available match to "import volume." It also refuses to overwrite the
existing data/raw/portwatch_baltimore_monthly.csv if a fresh pull disagrees
with it beyond a wide tolerance, since that file is the frozen snapshot every
number in the manuscript was verified against (see scripts/verify_claims.py).
If PortWatch's schema drifts further, that frozen file -- not a fresh pull --
remains the authoritative source; `python run_all.py --step 4` reproduces the
full analysis from it without needing to re-collect.
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"

import requests
import pandas as pd
import time

BASE = ("https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/"
        "services/Container_Metrics/FeatureServer/0/query")

PORTS = {"port103": "baltimore", "port180": "brunswick", "port2066": "port_virginia"}

# Fetch each port field in its OWN query rather than one combined multi-field
# query. The live service intermittently 400s on certain multi-field
# combinations (observed: any query whose outFields includes port180
# alongside another port field fails with "Invalid query parameters", while
# each field queried alone succeeds) -- isolating fields per-request means a
# problem with one port's field does not block collection of the others,
# which matters because only port103 (Baltimore) is required downstream.
port_frames = {}
for portid, label in PORTS.items():
    all_records = []
    offset = 0
    print(f"Fetching Container_Metrics — {label} ({portid}) ...")
    while True:
        params = {
            "where"             : "metric='import_container'",
            "outFields"         : f"date_in,{portid}",
            "resultOffset"      : str(offset),
            "resultRecordCount" : "2000",
            "f"                 : "json",
        }
        resp = requests.post(BASE, data=params, timeout=60)
        d = resp.json()

        if d.get("error"):
            print(f"  Error fetching {label}:", d["error"])
            break

        features = d.get("features", [])
        if not features:
            break

        all_records.extend([f["attributes"] for f in features])
        print(f"  offset={offset}  got={len(features)}  total={len(all_records)}")

        if not d.get("exceededTransferLimit", False):
            break
        offset += 2000
        time.sleep(0.3)

    if all_records:
        pdf = pd.DataFrame(all_records)
        pdf["date"] = pd.to_datetime(pdf["date_in"])
        pdf = pdf[["date", portid]].rename(columns={portid: label + "_import_vol"})
        port_frames[label] = pdf
    else:
        print(f"  WARNING: no data collected for {label} ({portid}) — skipping")

if "baltimore" not in port_frames:
    raise RuntimeError(
        "PortWatch Container_Metrics fetch failed for port103 (Baltimore) — "
        "this is the field 04_master_dataset.py requires. Aborting rather than "
        "silently writing an incomplete file."
    )

# Merge whichever port frames succeeded on date (outer join preserves full
# coverage even if a given port's field is temporarily unavailable)
df = port_frames["baltimore"]
for label, pdf in port_frames.items():
    if label == "baltimore":
        continue
    df = df.merge(pdf, on="date", how="outer")
df = df.sort_values("date").reset_index(drop=True)

print(f"\nDate range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Total rows: {len(df)}")
print(f"Missing values: {df.isna().sum().to_dict()}")

print("\n2022-2024 Baltimore data:")
preview_cols = ["date"] + [c for c in ["baltimore_import_vol", "brunswick_import_vol",
                                        "port_virginia_import_vol"] if c in df.columns]
balt_2024 = df[df["date"].dt.year >= 2022][preview_cols].copy()
print(balt_2024.to_string())

df.to_csv(str(DATA_RAW / "portwatch_container_metrics.csv"), index=False)
print("\nSaved: data/raw/portwatch_container_metrics.csv")

# Baltimore-only extract: this is the file 04_master_dataset.py actually
# consumes as the primary shock variable (balt_import_vol). Writing it here,
# from the same pull as the full container_metrics.csv above, keeps the
# from-scratch pipeline (python run_all.py) reproducible end to end --
# previously no script produced this file even though downstream steps
# (04_master_dataset.py, 06_safety_stock.py, 07_placebo_robustness.py)
# required it, which would crash a clean-checkout run at Step 4.
balt_only = df[["date", "baltimore_import_vol"]].dropna().sort_values("date").reset_index(drop=True)
balt_only = balt_only.drop_duplicates(subset="date", keep="first")

target = DATA_RAW / "portwatch_baltimore_monthly.csv"
if target.exists():
    existing = pd.read_csv(target, parse_dates=["date"])
    common = existing.merge(balt_only, on="date", suffixes=("_old", "_new"))
    if len(common) > 0:
        rel_diff = ((common["baltimore_import_vol_new"] - common["baltimore_import_vol_old"]).abs()
                    / common["baltimore_import_vol_old"].abs().clip(lower=1)).median()
        if rel_diff > 0.05 or abs(len(balt_only) - len(existing)) > 5:
            print(f"\n  WARNING: fresh pull disagrees with the existing frozen "
                  f"{target.name} (median relative diff={rel_diff:.1%}, "
                  f"row counts {len(balt_only)} vs {len(existing)}).")
            print("  NOT overwriting the frozen file -- it is the snapshot every")
            print("  number in the manuscript was verified against. Fresh pull")
            print("  saved separately for inspection at "
                  f"{target.with_name('portwatch_baltimore_monthly_FRESH_PULL.csv').name}.")
            balt_only.to_csv(target.with_name("portwatch_baltimore_monthly_FRESH_PULL.csv"), index=False)
            print("Step 2b complete (frozen file preserved) ✓")
            raise SystemExit(0)
        else:
            print(f"\n  Fresh pull matches existing frozen file within tolerance "
                  f"(median relative diff={rel_diff:.2%}) -- safe to overwrite.")

balt_only.to_csv(target, index=False)
print(f"Saved: data/raw/portwatch_baltimore_monthly.csv ({len(balt_only)} rows)")
print("Step 2b complete ✓")
