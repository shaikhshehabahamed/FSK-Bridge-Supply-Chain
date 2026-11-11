"""
Step 8c: Event-Study Specification
====================================
Estimates separate coefficients for each month relative to the shock.
Reference month: February 2024 (t = -1, one month before collapse).

Model:
  Y_t = α + β₁·t + Σ_{k≠-1} δₖ·D_{t,k} + γ₁·covid + γ₂·uaw + Σ month_FE + ε_t

  D_{t,k} = 1 if observation t is exactly k months from March 2024
  k ranges from -12 (Mar 2023) to +9 (Dec 2024)
  k = -1 (Feb 2024) is the omitted reference category

Pre-period coefficients (k < 0) should be flat/near zero → validates ITS assumption.
Post-period coefficients (k ≥ 0) show the dynamic treatment path.
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"



import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pickle, warnings
warnings.filterwarnings("ignore")

master = pd.read_csv(str(DATA_PROC / "master.csv"),
                     parse_dates=["date"], index_col="date")

SHOCK_DATE  = pd.Timestamp("2024-03-01")   # t = 0
REF_DATE    = pd.Timestamp("2024-02-01")   # t = -1 (omitted reference)
MONTH_DUMMIES = [f"m{m:02d}" for m in range(2, 13)]

# ── Build event-time relative indicators ────────────────────────────────────
shock_row = master.index.get_loc(SHOCK_DATE)
master["event_k"] = np.nan
for i, dt in enumerate(master.index):
    diff_months = (dt.year - SHOCK_DATE.year)*12 + (dt.month - SHOCK_DATE.month)
    master.loc[dt, "event_k"] = diff_months

# Range: k = -12 to +9
K_MIN, K_MAX = -12, 9
K_REF = -1   # omitted

k_values = [k for k in range(K_MIN, K_MAX+1) if k != K_REF]

def build_event_X(df):
    """Build design matrix with event-time dummies.

    uaw_strike (Oct-Nov 2023) is deliberately EXCLUDED here even though the
    ITS models (05_its_model.py) include it as a control. Within this event
    window (k=-12..+9 = Mar 2023..Dec 2024), Oct/Nov 2023 already have their
    own singleton event-time dummies (ev_-05, ev_-04), so uaw_strike would be
    an exact linear combination of those two columns (uaw_strike = ev_-05 +
    ev_-04), making the design matrix rank-deficient. statsmodels silently
    fits this via a pseudo-inverse with no warning, which non-uniquely
    identifies uaw_strike, ev_-05, and ev_-04 and corrupts those two months'
    coefficients. covid_shock (Apr-Jun 2020) is unaffected: it falls entirely
    outside this event window and has no corresponding event-time dummy.
    """
    cols = ["t", "covid_shock"] + MONTH_DUMMIES
    Xbase = sm.add_constant(df[cols].copy())
    cn = ["const"] + cols

    for k in k_values:
        col = f"ev_{k:+03d}"
        Xbase[col] = (df["event_k"] == k).astype(int)
        cn.append(col)

    return Xbase, cn

OUTCOMES = {
    "ln_auto_prod"   : ("ln(Auto Production — DAUPSA)",  "#8e44ad"),
    "ppi_mv_parts"   : ("PPI: Motor Vehicle Parts",      "#e67e22"),
    "ln_hs8708_cif_B": ("ln(HS 8708 CIF Imports)",       "#27ae60"),
    "ppi_petrochem"  : ("PPI: Petrochem [PLACEBO]",      "#95a5a6"),
}

ev_results = {}

print("=" * 70)
print("EVENT-STUDY ESTIMATES  (reference: Feb 2024, k = -1)")
print("HAC(4) standard errors")
print("=" * 70)

for outcome, (label, col) in OUTCOMES.items():
    y   = master[outcome].dropna()
    Xdf, cn = build_event_X(master.loc[y.index])

    # Guard against silent rank-deficiency (statsmodels fits a deficient
    # design matrix via pseudo-inverse with no warning, which silently
    # non-uniquely identifies whichever columns are collinear).
    rank = np.linalg.matrix_rank(Xdf.values)
    if rank < Xdf.shape[1]:
        raise RuntimeError(
            f"Event-study design matrix for {outcome} is rank-deficient "
            f"({rank} < {Xdf.shape[1]} columns) -- coefficients would be "
            f"non-uniquely identified. Check for controls that exactly "
            f"duplicate a combination of event-time dummies."
        )

    res = sm.OLS(y.values, Xdf.values).fit(
        cov_type="HAC", cov_kwds={"maxlags":4,"use_correction":True})

    # Extract event-k coefficients
    ev_coefs = {}
    for k in k_values:
        col_name = f"ev_{k:+03d}"
        idx = cn.index(col_name)
        ev_coefs[k] = {
            "est":  res.params[idx],
            "se":   res.bse[idx],
            "lo":   res.conf_int()[idx, 0],
            "hi":   res.conf_int()[idx, 1],
            "pval": res.pvalues[idx],
        }
    # Reference point (k = -1) is identically 0 by construction
    ev_coefs[K_REF] = {"est":0,"se":0,"lo":0,"hi":0,"pval":1.0}

    ev_results[outcome] = {
        "label": label, "color": col,
        "coefs": ev_coefs, "r2": res.rsquared_adj,
        "res": res,
    }

    print(f"\n── {label}  (adj-R²={res.rsquared_adj:.3f})")
    print(f"  {'k':>5}  {'date':>10}  {'est':>9}  {'se':>8}  {'p':>7}")
    for k in sorted(ev_coefs):
        dt = SHOCK_DATE + pd.DateOffset(months=k)
        e  = ev_coefs[k]
        sig = "***" if e['pval']<0.001 else "**" if e['pval']<0.01 \
              else "*" if e['pval']<0.05 else ""
        marker = " ← SHOCK" if k==0 else " ← ref" if k==-1 else ""
        print(f"  {k:>5}  {str(dt.date()):>10}  {e['est']:>+9.4f}  "
              f"{e['se']:>8.4f}  {e['pval']:>7.4f}{sig}{marker}")

# ── PRE-TREND JOINT F-TEST (k = -12 to -2 should be jointly zero) ───────────
# This is the formal statistical test of the ITS/event-study identifying
# assumption (flat pre-trend). It is saved below (not just printed) because
# it fails for two of the four outcomes -- ln_auto_prod (the PRIMARY outcome)
# and ppi_petrochem (the PLACEBO) -- which is a first-order result for
# interpreting the paper's causal claims, not a cosmetic diagnostic. See
# Table 7b and the manuscript's Limitations section: the auto-production
# failure corroborates (with a formal test, not just visual judgment) why
# 05b_its_plots.py already treats the ITS counterfactual for ln_auto_prod as
# unreliable and substitutes a YoY comparison; the placebo failure means the
# "no pre-trend" falsification check itself is not clean and must be
# disclosed, not silently omitted.
print("\n" + "=" * 70)
print("PRE-TREND JOINT F-TEST  H₀: all δₖ = 0 for k ∈ [-12, -2]")
print("=" * 70)

pretrend_tests = {}
for outcome, data in ev_results.items():
    res = data["res"]
    Xdf, cn = build_event_X(master.loc[master[outcome].dropna().index])

    pre_k_cols = [f"ev_{k:+03d}" for k in range(K_MIN, -1)]   # k=-12 to k=-2
    R = np.zeros((len(pre_k_cols), len(cn)))
    for i, col_name in enumerate(pre_k_cols):
        if col_name in cn:
            R[i, cn.index(col_name)] = 1

    ftest = res.f_test(R)
    fval  = float(ftest.fvalue)
    pval  = float(ftest.pvalue)
    flat  = pval > 0.10
    sig   = "✓ flat (good)" if flat else "⚠ not flat"
    pretrend_tests[outcome] = {"label": data["label"], "F": fval, "p": pval, "flat": flat}
    print(f"  {data['label'][:45]:45s}  F={fval:.2f}  p={pval:.3f}  {sig}")

n_not_flat = sum(1 for v in pretrend_tests.values() if not v["flat"])
print(f"\n  {n_not_flat}/{len(pretrend_tests)} series fail the flat-pre-trend test at the 10% level.")
print("  This must be disclosed as a limitation, not just diagnosed and dropped:")
print("  it directly affects how much weight the ITS point estimates for those")
print("  series (auto production and the petrochem placebo) can bear.")

# ── Save results ─────────────────────────────────────────────────────────────
# Strip non-serialisable res object before pickling
ev_save = {k: {kk: vv for kk, vv in v.items() if kk != "res"}
           for k, v in ev_results.items()}

with open(str(DATA_PROC / "event_study.pkl"),"wb") as f:
    pickle.dump({"outcomes": ev_save, "pretrend_tests": pretrend_tests}, f)

print("\nStep 8c — Event-study complete ✓")
