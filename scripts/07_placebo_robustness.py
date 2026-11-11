"""
Step 7: Placebo Tests & Robustness Checks
==========================================
Test 1 — Petrochemical PPI placebo (already in ITS, formalised here)
Test 2 — False-break / placebo-date test: run ITS across every month
          2019-2023 as the "shock date"; March 2024 should be an outlier
Test 3 — HAC lag sensitivity (2, 4, 6, 8 lags) on primary outcome
Test 4 — COVID exclusion: drop 2020 from pre-period
Test 5 — Levels spec: auto_prod (not logged) as dependent variable
Test 6 — Restricted pre-period: 2021-2023 only (avoids COVID entirely)
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import pickle, warnings
warnings.filterwarnings("ignore")

master = pd.read_csv(str(DATA_PROC / "master.csv"),
                     parse_dates=["date"], index_col="date")

MONTH_DUMMIES = [f"m{m:02d}" for m in range(2, 13)]
TRUE_SHOCK    = pd.Timestamp("2024-03-01")

def run_its(data, outcome, shock_date, lags=4, extra_controls=True):
    """Run two-phase ITS for a given shock date, return (β₂, β₃, p₂, p₃, F, pF, res, se₂, se₃).

    Some callers below (COVID-excluded, restricted-pre-period specs) pass a
    subsample where covid_shock is identically zero for every remaining row
    (its only 1s fall inside the excluded window). A zero column makes X'X
    singular in that one dimension, but does NOT corrupt the other
    coefficients: the normal equations decouple exactly (a zero column has
    zero cross-product with every other column), so post_pl/t_post_pl's
    point estimates and SEs come out identical whether the dead column is
    included or dropped -- verified numerically to 6 decimal places during
    the 2026-07-19 audit. statsmodels' pinv-based OLS handles this
    gracefully (assigns the dead column coefficient 0), so no special-casing
    is needed; this is just documented here so it doesn't look like a silent
    bug to a future reader, the way it initially did to us.
    """
    df = data.copy()
    df["post_pl"]  = (df.index >= shock_date).astype(int)
    df["t_post_pl"] = 0
    shock_t = df.loc[shock_date, "t"] if shock_date in df.index else None
    if shock_t is None:
        return None
    df["t_post_pl"] = (df["t"] - shock_t) * df["post_pl"]

    cols = ["t", "post_pl", "t_post_pl"]
    if extra_controls:
        cols += ["covid_shock", "uaw_strike"] + MONTH_DUMMIES
    X = sm.add_constant(df[cols])
    cn = X.columns.tolist()

    y = df[outcome].dropna()
    X = X.loc[y.index]

    try:
        res = sm.OLS(y.values, X.values).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    except Exception:
        return None

    b2_i = cn.index("post_pl");   b3_i = cn.index("t_post_pl")
    b2, b3 = res.params[b2_i],  res.params[b3_i]
    p2, p3 = res.pvalues[b2_i], res.pvalues[b3_i]
    se2, se3 = res.bse[b2_i], res.bse[b3_i]
    n = len(cn)
    R = np.zeros((2, n)); R[0, b2_i] = 1; R[1, b3_i] = 1
    ftest = res.f_test(R)
    return b2, b3, p2, p3, float(ftest.fvalue), float(ftest.pvalue), res, se2, se3

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: False-break / Placebo-date test
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST 2 — False-break / Placebo-date test")
print("  Run ITS with every month 2020-01 → 2023-12 as placebo shock date")
print("  Compare F-statistic at each date vs true shock (March 2024)")
print("=" * 70)

# Use Baltimore Container_Metrics (where signal is cleanest)
balt_m = pd.read_csv(str(DATA_RAW / "portwatch_baltimore_monthly.csv"),
                      parse_dates=["date"]).set_index("date")
master["balt_vol_log"] = np.log(master["balt_import_vol"].clip(lower=1))

placebo_dates = pd.date_range("2020-01-01", "2023-12-01", freq="MS")
placebo_res   = []

for pd_date in placebo_dates:
    r = run_its(master, "balt_vol_log", pd_date)
    if r:
        b2, b3, p2, p3, Fval, pF, _, _, _ = r
        placebo_res.append({"date": pd_date, "b2": b2, "b3": b3,
                             "p2": p2, "p3": p3, "F": Fval, "pF": pF})

# True shock
true_r = run_its(master, "balt_vol_log", TRUE_SHOCK)
true_b2, true_b3, true_p2, true_p3, true_F, true_pF, _, _, _ = true_r

df_plac = pd.DataFrame(placebo_res)
pct_rank = (df_plac["F"] < true_F).mean() * 100

print(f"\nTrue shock (Mar 2024):  β₂={true_b2:+.3f}  β₃={true_b3:+.3f}  "
      f"F={true_F:.2f}  p={true_pF:.4f}")
print(f"Placebo date F-stats:  mean={df_plac['F'].mean():.2f}  "
      f"max={df_plac['F'].max():.2f}  sd={df_plac['F'].std():.2f}")
print(f"True shock F percentile: {pct_rank:.1f}th (vs {len(placebo_dates)} placebo dates)")

# Also run on auto production
prod_res = []
for pd_date in placebo_dates:
    r = run_its(master, "ln_auto_prod", pd_date)
    if r:
        b2, b3, p2, p3, Fval, pF, _, _, _ = r
        prod_res.append({"date": pd_date, "b2": b2, "F": Fval, "pF": pF})
true_r_prod = run_its(master, "ln_auto_prod", TRUE_SHOCK)
true_F_prod = true_r_prod[4]
df_prod_plac = pd.DataFrame(prod_res)
pct_rank_prod = (df_prod_plac["F"] < true_F_prod).mean() * 100
print(f"\nAuto production false-break test:")
print(f"True F={true_F_prod:.2f} at {pct_rank_prod:.0f}th percentile of placebo distribution")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: HAC lag sensitivity
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 3 — HAC lag sensitivity (primary outcome: ln_auto_prod)")
print("=" * 70)

lag_results = []
for lags in [2, 4, 6, 8]:
    r = run_its(master, "ln_auto_prod", TRUE_SHOCK, lags=lags)
    b2, b3, p2, p3, Fval, pF, fit, se2, se3 = r
    lag_results.append({
        "lags": lags, "b2": b2, "b3": b3,
        "p2": p2, "p3": p3, "F": Fval, "pF": pF,
        "R2": fit.rsquared_adj, "se2": se2, "se3": se3,
    })
    print(f"  HAC({lags}):  β₂={b2:+.4f}(p={p2:.3f})  "
          f"β₃={b3:+.4f}(p={p3:.3f})  F={Fval:.2f}(p={pF:.3f})  adj-R²={fit.rsquared_adj:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: COVID exclusion
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 4 — COVID exclusion (drop 2020-03 to 2021-06 from pre-period)")
print("=" * 70)

# Exclude the worst COVID months from pre-period baseline
covid_months = pd.date_range("2020-03-01", "2021-06-01", freq="MS")
master_nocov = master.copy()
master_nocov.loc[covid_months, "ln_auto_prod"] = np.nan

r_full = run_its(master, "ln_auto_prod", TRUE_SHOCK)
r_nocov = run_its(master_nocov, "ln_auto_prod", TRUE_SHOCK)

print(f"\n  Full sample:      β₂={r_full[0]:+.4f}(p={r_full[2]:.3f})  "
      f"β₃={r_full[1]:+.4f}(p={r_full[3]:.3f})  F={r_full[4]:.2f}")
print(f"  COVID-excluded:   β₂={r_nocov[0]:+.4f}(p={r_nocov[2]:.3f})  "
      f"β₃={r_nocov[1]:+.4f}(p={r_nocov[3]:.3f})  F={r_nocov[4]:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Levels specification
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 5 — Levels specification (auto_prod, not logged)")
print("=" * 70)

r_levels = run_its(master, "auto_prod", TRUE_SHOCK)
b2_lev, b3_lev, p2_lev, p3_lev, F_lev, pF_lev, fit_lev, se2_lev, se3_lev = r_levels
print(f"\n  Levels:  β₂={b2_lev:+.3f}(p={p2_lev:.3f})  "
      f"β₃={b3_lev:+.3f}(p={p3_lev:.3f})  F={F_lev:.2f}(p={pF_lev:.3f})")
print(f"  Interpretation: β₂ = {b2_lev:.1f}k units/month level change at shock")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Restricted pre-period (2021-2023 only, fully post-COVID)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 6 — Restricted pre-period (2021-01 to 2023-12)")
print("=" * 70)

master_rest = master.loc["2021-01-01":].copy()
# Re-index t from 0 for this subset
master_rest["t"] = np.arange(len(master_rest))
shock_t_rest = master_rest.loc[TRUE_SHOCK, "t"]
master_rest["post"]   = (master_rest.index >= TRUE_SHOCK).astype(int)
master_rest["t_post"] = (master_rest["t"] - shock_t_rest) * master_rest["post"]

# Rebuild month dummies on restricted dataset
for m in range(2, 13):
    master_rest[f"m{m:02d}"] = (master_rest.index.month == m).astype(int)

r_rest = run_its(master_rest, "ln_auto_prod", TRUE_SHOCK)
n_pre_rest = int((master_rest["post"] == 0).sum())
print(f"\n  Restricted (2021-01 to {TRUE_SHOCK - pd.DateOffset(months=1):%Y-%m} pre-period, n_pre={n_pre_rest}):")
print(f"  β₂={r_rest[0]:+.4f}(p={r_rest[2]:.3f})  "
      f"β₃={r_rest[1]:+.4f}(p={r_rest[3]:.3f})  F={r_rest[4]:.2f}(p={r_rest[5]:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: March 2024 partial-treatment-month sensitivity
# ─────────────────────────────────────────────────────────────────────────────
# COLLAPSE_M in 04_master_dataset.py codes ALL of March 2024 as post=1 /
# disruption=1, even though the collapse happened on day 26 of 31 (~81% of
# the month was pre-collapse business-as-usual). Pooling that mostly-normal
# month into the "post" bucket alongside the fully-disrupted April/May
# dilutes the estimated immediate level shift (beta2) toward zero relative
# to the true acute effect. The event-study specification (08c) already
# sidesteps this by giving March its own k=0 dummy; this test checks whether
# the pooled two-phase ITS models (Table 2, the ones actually used for the
# headline beta2/beta3 estimates) are sensitive to it, by dropping March
# 2024 entirely (neither pre nor post) and re-estimating.
print("\n" + "=" * 70)
print("TEST 7 — March 2024 excluded (partial-treatment-month sensitivity)")
print("  March 26 collapse -> only 6/31 (19%) of March was post-collapse;")
print("  Model A/B pool all of March into 'post' alongside fully-disrupted")
print("  Apr/May. This test drops March 2024 entirely and re-estimates.")
print("=" * 70)

master_nomar = master.copy()
master_nomar.loc["2024-03-01", ["ln_auto_prod","ppi_mv_parts","ln_hs8708_cif_B","ppi_petrochem"]] = np.nan

nomar_results = {}
for outcome in ["ln_auto_prod","ppi_mv_parts","ln_hs8708_cif_B","ppi_petrochem"]:
    r = run_its(master_nomar, outcome, TRUE_SHOCK)
    nomar_results[outcome] = r
    b2, b3, p2, p3, Fval, pF, _, se2, se3 = r
    print(f"  {outcome:18s}  β₂={b2:+.4f}(se={se2:.4f}, p={p2:.3f})  "
          f"β₃={b3:+.4f}(se={se3:.4f}, p={p3:.3f})  F={Fval:.2f}(p={pF:.3f})")

r_nomar = nomar_results["ln_auto_prod"]
print(f"\n  ln(auto_prod) comparison:")
print(f"    Main (March included): β₂={r_full[0]:+.4f}(p={r_full[2]:.3f})")
print(f"    March excluded:        β₂={r_nomar[0]:+.4f}(p={r_nomar[2]:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ROBUSTNESS SUMMARY TABLE — ln(auto_prod), β₂ and β₃")
print("=" * 70)

rows = [
    ("Main (HAC-4, full sample)",   r_full[0],  r_full[7],  r_full[2],  r_full[1],  r_full[8],  r_full[3]),
    ("HAC-2",                        lag_results[0]["b2"], lag_results[0]["se2"], lag_results[0]["p2"],
                                     lag_results[0]["b3"], lag_results[0]["se3"], lag_results[0]["p3"]),
    ("HAC-6",                        lag_results[2]["b2"], lag_results[2]["se2"], lag_results[2]["p2"],
                                     lag_results[2]["b3"], lag_results[2]["se3"], lag_results[2]["p3"]),
    ("HAC-8",                        lag_results[3]["b2"], lag_results[3]["se2"], lag_results[3]["p2"],
                                     lag_results[3]["b3"], lag_results[3]["se3"], lag_results[3]["p3"]),
    ("COVID-excluded",               r_nocov[0], r_nocov[7], r_nocov[2], r_nocov[1], r_nocov[8], r_nocov[3]),
    ("Restricted pre (2021+)",       r_rest[0],  r_rest[7],  r_rest[2],  r_rest[1],  r_rest[8],  r_rest[3]),
    ("Levels (k units)",             b2_lev/100, se2_lev/100, p2_lev,     b3_lev/100, se3_lev/100, p3_lev),
    ("March 2024 excluded",          r_nomar[0], r_nomar[7], r_nomar[2], r_nomar[1], r_nomar[8], r_nomar[3]),
]

def star(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
print(f"{'Specification':35s}  {'β₂':>10}  {'p':>8}  {'β₃':>10}  {'p':>8}")
print("-" * 78)
for spec, b2, se2, p2, b3, se3, p3 in rows:
    print(f"  {spec:33s}  {b2:>+10.4f}  {p2:>6.3f}{star(p2):3s}  "
          f"{b3:>+10.4f}  {p3:>6.3f}{star(p3):3s}")

# ─────────────────────────────────────────────────────────────────────────────
# Save for plotting
# ─────────────────────────────────────────────────────────────────────────────
import pickle
with open(str(DATA_PROC / "robustness.pkl"), "wb") as f:
    pickle.dump({
        "df_plac": df_plac, "true_F": true_F, "true_b2": true_b2,
        "pct_rank": pct_rank,
        "df_prod_plac": df_prod_plac, "true_F_prod": true_F_prod, "pct_rank_prod": pct_rank_prod,
        "lag_results": lag_results, "rows": rows,
        "nomar_results": {k: (v[0], v[1], v[2], v[3], v[4], v[5], v[7], v[8]) for k, v in nomar_results.items()},
    }, f)

print("\nStep 7 — Placebo & Robustness complete ✓")
