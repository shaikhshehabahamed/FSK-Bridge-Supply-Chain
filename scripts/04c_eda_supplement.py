"""
Step 4c: EDA Supplement
Pre-trend analysis + correlation heatmap + distributional plots.
Tests the parallel-trends assumption needed for the ITS model.
Figure S1 (supplementary).
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

import _style
from _style import COLOR, INK_PRIMARY, INK_SECONDARY, RED, BLUE, save_figure, style_axes, event_lines
_style.apply_style()

master = pd.read_csv(str(DATA_PROC / "master.csv"),
                     parse_dates=["date"], index_col="date")

PRE = master[master["post"] == 0].copy()

# ─────────────────────────────────────────────────────────────────────────────
# PRE-TREND TEST: OLS on pre-period only
# Key question: is there a pre-existing trend in auto_prod that could
# confound the ITS break? If trend is flat/stable, ITS assumption holds.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PRE-TREND ANALYSIS (pre-period OLS, t = month index)")
print("=" * 70)

KEY_OUTCOMES = {
    "auto_prod"      : "Auto Production (DAUPSA)",
    "ppi_mv_parts"   : "PPI: MV Parts Mfg",
    "balt_import_vol": "Baltimore Import Vol",
    "hs8708_cif_B"   : "HS 8708 CIF Imports",
    "ppi_petrochem"  : "PPI Petrochem (PLACEBO)",
}

pre_trends = {}
for col, label in KEY_OUTCOMES.items():
    y = PRE[col].dropna()
    x = PRE.loc[y.index, "t"].values
    slope, intercept, r, p, se = stats.linregress(x, y.values)
    pct_per_month = (slope / y.mean()) * 100
    pre_trends[col] = {"label": label, "slope": slope, "pct_pm": pct_per_month,
                        "r2": r**2, "p": p, "n": len(y)}
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "(ns)"
    print(f"  {label[:40]:40s}  slope={slope:+.4f}  pct/mo={pct_per_month:+.3f}%  "
          f"R²={r**2:.3f}  p={p:.4f} {sig}")

# ─────────────────────────────────────────────────────────────────────────────
# YoY DECOMPOSITION: How much of the post-shock change is the disruption vs trend?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("YoY DECOMPOSITION — Auto Production (DAUPSA)")
print("=" * 70)

prod = master["auto_prod"]
for m in range(3, 9):  # March through August 2024
    dt24 = pd.Timestamp(f"2024-{m:02d}-01")
    dt23 = pd.Timestamp(f"2023-{m:02d}-01")
    v24 = prod.loc[dt24]; v23 = prod.loc[dt23]
    print(f"  {dt24.strftime('%b %Y')}: {v24:.1f}k  (vs {v23:.1f}k in 2023)  "
          f"YoY={((v24/v23)-1)*100:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE S1: EDA supplement — 3-panel
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11.5, 4.2))
gs = gridspec.GridSpec(1, 3, wspace=0.42)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

COLLAPSE = pd.Timestamp("2024-03-26")
FULL_OP  = pd.Timestamp("2024-06-10")

# ── Panel A: Pre-period trend in auto production ────────────────────────────
ax = axes[0]
y = master["auto_prod"]
pre_y = PRE["auto_prod"]
pre_x = PRE["t"]

slope, intercept, r_pt, p_pt, se_pt = stats.linregress(pre_x, pre_y)

ax.scatter(master.index, y, color=COLOR["auto_prod"], s=13, alpha=0.6, zorder=3)
ax.plot(PRE.index, intercept + slope * PRE["t"], color=COLOR["shock"], lw=1.8,
        ls="-", label="Pre-period trend", zorder=4)
POST_IDX = master[master["post"] == 1].index
ax.plot(POST_IDX, intercept + slope * master.loc[POST_IDX, "t"], color=COLOR["shock"],
        lw=1.6, ls="--", alpha=0.6, label="Counterfactual", zorder=4)

ax.axvline(COLLAPSE, color=COLOR["shock"], lw=1.3, ls="--", alpha=0.85)
ax.axvspan(COLLAPSE, FULL_OP, color=COLOR["shock"], alpha=0.06)

ax.set_title("(A)  Auto production: pre-trend\n& counterfactual", fontsize=_style.FS_TITLE, pad=6)
ax.set_ylabel("Production (DAUPSA, k units/mo)", fontsize=_style.FS_AXIS)
ax.legend(fontsize=_style.FS_FOOTNOTE, loc="lower left")
# The pre-period trend is in fact highly statistically significant
# (p=0.0001, R^2=0.24) -- a linear fit through the COVID crash-and-recovery
# period, not noise. This is exactly why the linear ITS counterfactual for
# ln(auto_prod) is flagged elsewhere (05b_its_plots.py, Table 2) as unreliable
# for extrapolation: the "significant slope" is a COVID-era artifact, not a
# stable secular trend a short post-period extrapolation can trust.
sig_pt = "***" if p_pt < 0.001 else "**" if p_pt < 0.01 else "*" if p_pt < 0.05 else "(ns)"
ax.text(0.03, 0.97,
        f"Trend: {(slope/pre_y.mean())*100:+.2f}%/mo, p={p_pt:.4f} {sig_pt}\n"
        f"(significant — COVID-era artifact,\nnot a stable trend; see Fig. 2)",
        transform=ax.transAxes, fontsize=_style.FS_FOOTNOTE, va="top", color=INK_SECONDARY,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e1e0d9"))
style_axes(ax)

# ── Panel B: Correlation heatmap ────────────────────────────────────────────
# Two-hue diverging colormap (blue <-> red through a neutral gray midpoint)
# rather than a 3-hue rainbow (e.g. RdYlGn) -- a diverging encoding should
# never route through a third hue at the midpoint.
ax = axes[1]
cols = ["auto_prod", "ppi_mv_parts", "balt_import_vol", "hs8708_cif_B", "ppi_petrochem"]
labels = ["AutoProd", "PPI parts", "BaltPort", "HS8708", "Petrochem\n(placebo)"]
C = PRE[cols].corr().values

diverging = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, "#f0efec", RED])
im = ax.imshow(C, cmap=diverging, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=_style.FS_FOOTNOTE, rotation=35, ha="right")
ax.set_yticklabels(labels, fontsize=_style.FS_FOOTNOTE)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                fontsize=_style.FS_FOOTNOTE, color="white" if abs(C[i, j]) > 0.6 else INK_PRIMARY)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
cbar.ax.tick_params(labelsize=_style.FS_FOOTNOTE)
ax.set_title("(B)  Pre-period correlation matrix\n(n=62 months, 2019–2023)", fontsize=_style.FS_TITLE, pad=6)

# ── Panel C: Distribution shifts — auto_prod by phase ───────────────────────
ax = axes[2]
DISR = master[master["disruption"] == 1]
POST_r = master[(master["post"] == 1) & (master["disruption"] == 0)]

bins = np.linspace(0, 300, 22)
phase_colors = [COLOR["auto_prod"], COLOR["shock"], COLOR["recovery"]]
for data, lab, col in [(PRE["auto_prod"], "Pre (2019–2023)", phase_colors[0]),
                        (DISR["auto_prod"], "Disruption (Mar–May 2024)", phase_colors[1]),
                        (POST_r["auto_prod"], "Recovery (Jun–Dec 2024)", phase_colors[2])]:
    ax.hist(data.dropna(), bins=bins, alpha=0.55, color=col, label=lab, zorder=2)
    ax.axvline(data.mean(), color=col, lw=1.8, ls="--", zorder=3)

ax.set_title("(C)  Auto production distribution\nby phase", fontsize=_style.FS_TITLE, pad=6)
ax.set_xlabel("Production (DAUPSA, k units/mo)", fontsize=_style.FS_AXIS)
ax.set_ylabel("Frequency", fontsize=_style.FS_AXIS)
ax.legend(fontsize=_style.FS_FOOTNOTE)
style_axes(ax)

axes[0].xaxis.set_major_locator(mdates.YearLocator())
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout(rect=[0, 0, 1, 1])
save_figure(fig, "figureS1_pretrend_diagnostics", category="supplementary")
print("Step 4 complete")
