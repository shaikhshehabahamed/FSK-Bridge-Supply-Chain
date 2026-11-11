"""
Step 5: Interrupted Time Series — Core ITS Models
Newey-West HAC(4) standard errors throughout.
conf_int returns numpy array shape (n_params, 2).
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"



import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
import pickle, warnings
warnings.filterwarnings("ignore")

master = pd.read_csv(str(DATA_PROC / "master.csv"),
                     parse_dates=["date"], index_col="date")

MONTH_DUMMIES = [f"m{m:02d}" for m in range(2, 13)]

def build_X(df, phase="two"):
    if phase == "two":
        cols = ["t", "post", "t_post", "covid_shock", "uaw_strike"] + MONTH_DUMMIES
    else:
        cols = ["t", "disruption", "t_post", "recovery",
                "covid_shock", "uaw_strike"] + MONTH_DUMMIES
    return sm.add_constant(df[cols].copy()), ["const"] + cols

def hac_ols(y_arr, X_arr, lags=4):
    return sm.OLS(y_arr, X_arr).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})

def fmt_coef(res, col_names, key):
    idx = col_names.index(key)
    v   = res.params[idx]
    p   = res.pvalues[idx]
    ci  = res.conf_int()          # shape (n_params, 2)
    lo, hi = ci[idx, 0], ci[idx, 1]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "   "
    return f"{v:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}  (p={p:.3f})"

OUTCOMES = {
    "ln_auto_prod"   : "ln(Auto Production — DAUPSA)    [PRIMARY]",
    "ppi_mv_parts"   : "PPI: Motor Vehicle Parts        [PRICE]",
    "ln_hs8708_cif_B": "ln(HS 8708 CIF Imports)         [TRADE / NULL]",
    "ppi_petrochem"  : "PPI: Petrochemical Mfg          [PLACEBO]",
}

store_a, store_b = {}, {}

# ─────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("MODEL A — Two-phase ITS")
print("  Y_t = α + β₁t + β₂·post + β₃·t_post + controls + ε_t")
print("  SEs: Newey-West HAC (4 lags)   n=72 months (2019-01 – 2024-12)")
print("=" * 72)

for outcome, label in OUTCOMES.items():
    y    = master[outcome].dropna()
    Xdf, cn = build_X(master.loc[y.index], "two")
    fit  = hac_ols(y.values, Xdf.values)
    store_a[outcome] = (fit, cn, Xdf)

    dw  = durbin_watson(fit.resid)
    lb4 = acorr_ljungbox(fit.resid, lags=[4], return_df=True)["lb_pvalue"].iloc[0]
    lb8 = acorr_ljungbox(fit.resid, lags=[8], return_df=True)["lb_pvalue"].iloc[0]

    print(f"\n── {label}")
    print(f"   β₁  pre-period trend:    {fmt_coef(fit, cn, 't')}")
    print(f"   β₂  level shift @ shock: {fmt_coef(fit, cn, 'post')}")
    print(f"   β₃  slope change post:   {fmt_coef(fit, cn, 't_post')}")
    print(f"   R²={fit.rsquared:.3f}  adj-R²={fit.rsquared_adj:.3f}  "
          f"AIC={fit.aic:.1f}  DW={dw:.2f}  LjBox Q(4)p={lb4:.3f} Q(8)p={lb8:.3f}")

# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("MODEL B — Three-phase ITS  (disruption vs recovery)")
print("  Y_t = α + β₁t + β₂·disr + β₃·t_post + β₄·recov + controls + ε_t")
print("=" * 72)

for outcome, label in OUTCOMES.items():
    y     = master[outcome].dropna()
    Xdf3, cn3 = build_X(master.loc[y.index], "three")
    fit3  = hac_ols(y.values, Xdf3.values)
    store_b[outcome] = (fit3, cn3, Xdf3)

    print(f"\n── {label}")
    print(f"   β₁  pre-period trend:     {fmt_coef(fit3, cn3, 't')}")
    print(f"   β₂  disruption level:     {fmt_coef(fit3, cn3, 'disruption')}")
    print(f"   β₃  slope change post:    {fmt_coef(fit3, cn3, 't_post')}")
    print(f"   β₄  recovery level:       {fmt_coef(fit3, cn3, 'recovery')}")
    print(f"   R²={fit3.rsquared:.3f}  adj-R²={fit3.rsquared_adj:.3f}  AIC={fit3.aic:.1f}")

# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("JOINT F-TESTS  H₀: β₂ = β₃ = 0  (Model A)")
print("=" * 72)

for outcome, label in OUTCOMES.items():
    fit, cn, _ = store_a[outcome]
    n = len(cn)
    R = np.zeros((2, n))
    R[0, cn.index("post")]   = 1
    R[1, cn.index("t_post")] = 1
    ftest = fit.f_test(R)
    fval  = float(ftest.fvalue)
    pval  = float(ftest.pvalue)
    sig   = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {label[:55]:55s}  F={fval:.2f}  p={pval:.4f} {sig}")

# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("COUNTERFACTUAL GAPS — ln(auto_prod) Model A")
print("Production under observed path vs no-shock counterfactual")
print("=" * 72)

fit, cn, Xdf_full = store_a["ln_auto_prod"]
params = fit.params

cf_rows = []
for dt in ["2024-03-01","2024-04-01","2024-05-01",
           "2024-06-01","2024-07-01","2024-08-01"]:
    if dt not in master.index:
        continue
    row_idx = master.index.get_loc(dt)
    Xrow = Xdf_full.values[row_idx, :].copy()

    # Counterfactual: zero out post and t_post
    Xcf  = Xrow.copy()
    Xcf[cn.index("post")]   = 0
    Xcf[cn.index("t_post")] = 0

    pred_actual = float(Xrow @ params)
    pred_cf     = float(Xcf  @ params)

    obs_log    = master.loc[dt, "ln_auto_prod"]
    # DAUPSA is "Thousands of Units, Seasonally Adjusted" (monthly, not SAAR):
    # convert log-values to actual monthly car counts via exp(log) * 1,000
    prod_obs   = np.exp(obs_log)  * 1_000
    prod_cf    = np.exp(pred_cf)  * 1_000
    gap_pct    = (prod_obs / prod_cf - 1) * 100
    gap_abs    = prod_obs - prod_cf

    cf_rows.append({
        "date": dt, "actual_monthly_cars": round(prod_obs),
        "counterfactual_monthly_cars": round(prod_cf),
        "gap_pct": round(gap_pct, 2), "gap_abs_cars": round(gap_abs)
    })
    print(f"  {dt}:  actual={prod_obs:8.0f} cars/mo  cf={prod_cf:8.0f} cars/mo  "
          f"gap={gap_pct:+5.1f}%")

# Save counterfactual table
pd.DataFrame(cf_rows).to_csv(
    str(DATA_PROC / "counterfactual_gaps.csv"), index=False)

# ─────────────────────────────────────────────────────────────────────────
with open(str(DATA_PROC / "its_results.pkl"),"wb") as f:
    pickle.dump({"model_a": store_a, "model_b": store_b}, f)

print("\nSaved → its_results.pkl + counterfactual_gaps.csv")
print("Step 5 — ITS models complete ✓")
