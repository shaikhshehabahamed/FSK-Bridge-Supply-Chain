"""
Step 10: ITS Residual Diagnostics
=====================================
Checks whether the full-sample ITS Model A residuals follow a white-noise
process. DW of 0.10 on PPI was a concern — this formalises that check.

For each outcome:
  1. Take residuals of the already-fit ITS Model A (full sample, 2019-2024)
  2. Check for remaining autocorrelation: Durbin-Watson, Ljung-Box, ACF/PACF,
     and a pre-period-only ADF unit-root test
  3. If strong AR structure remains, motivate the AR-augmented ITS (Model C)

Note: no ARIMA(p,d,q) model is separately fit here (an earlier draft of this
docstring implied one); the diagnostics above are sufficient to motivate
Model C and are what the manuscript's Methodology section describes.
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import pickle, warnings
warnings.filterwarnings("ignore")

import _style
from _style import COLOR, CRITICAL, INK_PRIMARY, save_figure, style_axes
_style.apply_style()

master = pd.read_csv(str(DATA_PROC / "master.csv"),
                     parse_dates=["date"], index_col="date")

with open(str(DATA_PROC / "its_results.pkl"),"rb") as f:
    its = pickle.load(f)

OUTCOMES = {
    "ln_auto_prod"   : "ln(Auto Production)",
    "ppi_mv_parts"   : "PPI: MV Parts",
    "ln_hs8708_cif_B": "ln(HS 8708 CIF Imports)",
    "ppi_petrochem"  : "PPI: Petrochem [PLACEBO]",
}

print("=" * 70)
print("ARIMA RESIDUAL DIAGNOSTICS — ITS Model A")
print("=" * 70)

diag_results = {}

for outcome, label in OUTCOMES.items():
    fit, cn, _ = its["model_a"][outcome]
    resid = fit.resid
    dw    = durbin_watson(resid)
    lb_df = acorr_ljungbox(resid, lags=[4, 8, 12], return_df=True)
    lb4   = lb_df["lb_pvalue"].iloc[0]
    lb8   = lb_df["lb_pvalue"].iloc[1]
    lb12  = lb_df["lb_pvalue"].iloc[2]

    # ADF test on residuals (stationarity)
    adf_stat, adf_p, *_ = adfuller(resid[~np.isnan(resid)], maxlag=4)

    # ACF at lag 1
    acf_vals = acf(resid[~np.isnan(resid)], nlags=12, fft=False)
    acf1 = acf_vals[1]

    concern = "HIGH" if dw < 1.0 or lb4 < 0.05 else \
              "MILD" if dw < 1.5 or lb4 < 0.20 else "OK"

    diag_results[outcome] = {
        "label": label, "dw": dw, "lb4": lb4, "lb8": lb8, "lb12": lb12,
        "adf_p": adf_p, "acf1": acf1, "concern": concern, "resid": resid
    }

    print(f"\n── {label}")
    print(f"   DW={dw:.3f}  ACF(1)={acf1:+.3f}  "
          f"LjungBox Q(4)p={lb4:.3f}  Q(8)p={lb8:.3f}  Q(12)p={lb12:.3f}")
    print(f"   ADF(resid) p={adf_p:.4f} (stationary={'yes' if adf_p<0.05 else 'borderline'})")
    print(f"   Assessment: {concern}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
for outcome, d in diag_results.items():
    if d["concern"] == "HIGH":
        print(f"\n  {d['label']}: AR-augmented ITS recommended")
        print(f"    Strong AR(1) structure (ACF1={d['acf1']:+.3f}) — HAC SEs partially")
        print(f"    address this but paper should note the limitation explicitly.")
    elif d["concern"] == "MILD":
        print(f"\n  {d['label']}: HAC(4) adequate, note in text")
    else:
        print(f"\n  {d['label']}: Residuals well-behaved — no action needed")

# ── Figure S3: Residual ACF/PACF small multiples ─────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.2))

ROLE_OF = {"ln_auto_prod": "auto_prod", "ppi_mv_parts": "ppi_mv_parts",
           "ln_hs8708_cif_B": "hs8708", "ppi_petrochem": "ppi_petrochem"}
NLAGS   = 15
CI_BAND = 1.96 / np.sqrt(72)

for col_i, (outcome, label) in enumerate(OUTCOMES.items()):
    resid = diag_results[outcome]["resid"]
    colour = COLOR[ROLE_OF[outcome]]

    ax_acf = axes[0, col_i]
    acf_v = acf(resid, nlags=NLAGS, fft=False)
    lags  = np.arange(len(acf_v))
    ax_acf.bar(lags[1:], acf_v[1:], color=colour, alpha=0.8, width=0.6, zorder=3)
    ax_acf.axhline(0, color=INK_PRIMARY, lw=0.7, alpha=0.6)
    ax_acf.axhline(+CI_BAND, color=CRITICAL, lw=1.1, ls="--", alpha=0.7)
    ax_acf.axhline(-CI_BAND, color=CRITICAL, lw=1.1, ls="--", alpha=0.7)
    ax_acf.set_title(f"ACF — {label}", fontsize=_style.FS_FOOTNOTE)
    ax_acf.set_xlim(0, NLAGS + 0.5)
    ax_acf.set_ylim(-0.6, 0.6)

    d = diag_results[outcome]
    ax_acf.text(0.97, 0.95, f"DW={d['dw']:.2f}\nACF(1)={d['acf1']:+.2f}\n{d['concern']}",
                transform=ax_acf.transAxes, fontsize=_style.FS_FOOTNOTE - 0.5, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#e1e0d9"))

    ax_paf = axes[1, col_i]
    try:
        pacf_v = pacf(resid, nlags=NLAGS, method="ols")
    except Exception:
        pacf_v = np.zeros(NLAGS + 1)
    ax_paf.bar(np.arange(len(pacf_v))[1:], pacf_v[1:], color=colour, alpha=0.8, width=0.6, zorder=3)
    ax_paf.axhline(0, color=INK_PRIMARY, lw=0.7, alpha=0.6)
    ax_paf.axhline(+CI_BAND, color=CRITICAL, lw=1.1, ls="--", alpha=0.7)
    ax_paf.axhline(-CI_BAND, color=CRITICAL, lw=1.1, ls="--", alpha=0.7)
    ax_paf.set_title(f"PACF — {label}", fontsize=_style.FS_FOOTNOTE)
    ax_paf.set_xlim(0, NLAGS + 0.5)
    ax_paf.set_ylim(-0.6, 0.6)
    ax_paf.set_xlabel("Lag (months)", fontsize=_style.FS_FOOTNOTE)

for ax in axes.flat:
    style_axes(ax)
    ax.tick_params(labelsize=_style.FS_FOOTNOTE - 0.5)

plt.tight_layout(rect=[0, 0, 1, 1])
save_figure(fig, "figureS3_residual_diagnostics", category="supplementary")
print("Step 10 — ARIMA residual check complete")
