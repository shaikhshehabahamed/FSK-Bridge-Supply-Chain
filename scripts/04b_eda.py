"""
Step 4b: Exploratory Data Analysis
Full EDA — descriptive stats, correlations, pre-trend tests,
and the four-panel 'story' plot that will anchor the paper's Figure 1.
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

import _style
from _style import COLOR, INK_SECONDARY, GRIDLINE, save_figure, style_axes, event_lines
_style.apply_style()

master = pd.read_csv(str(DATA_PROC / "master.csv"),
                     parse_dates=["date"], index_col="date")

PRE    = master[master["post"] == 0]
DISR   = master[master["disruption"] == 1]
POST   = master[(master["post"] == 1) & (master["disruption"] == 0)]

COLLAPSE_M = pd.Timestamp("2024-03-01")

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1: Descriptive statistics by phase
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("TABLE 1 — DESCRIPTIVE STATISTICS BY PHASE")
print("=" * 70)

KEY = {
    "auto_prod"      : "Auto Production (DAUPSA, k units/mo, SA)",
    "ppi_mv_parts"   : "PPI: MV Parts Mfg (Dec 2003=100)",
    "balt_import_vol": "Baltimore Import Vol (PortWatch)",
    "hs8708_cif_B"   : "HS 8708 CIF Imports ($B)",
    "vehicle_sales"  : "Total Vehicle Sales (M SAAR)",
}

for col, label in KEY.items():
    p_m  = PRE[col].mean();   p_s  = PRE[col].std()
    d_m  = DISR[col].mean();  d_s  = DISR[col].std()
    po_m = POST[col].mean();  po_s = POST[col].std()
    chg_disr = (d_m - p_m) / p_m * 100
    chg_post = (po_m - p_m) / p_m * 100
    print(f"\n  {label}")
    print(f"    Pre  (n={len(PRE)}):    mean={p_m:>12.3f}  sd={p_s:.3f}")
    print(f"    Disr (n={len(DISR)}):    mean={d_m:>12.3f}  sd={d_s:.3f}  Δ%={chg_disr:+.1f}%")
    print(f"    Post (n={len(POST)}):    mean={po_m:>12.3f}  sd={po_s:.3f}  Δ%={chg_post:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2: Welch t-tests (pre vs disruption, pre vs post)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 2 — WELCH T-TESTS (pre vs disruption | pre vs recovery)")
print("=" * 70)

for col, label in KEY.items():
    t1, p1 = stats.ttest_ind(PRE[col].dropna(), DISR[col].dropna(), equal_var=False)
    t2, p2 = stats.ttest_ind(PRE[col].dropna(), POST[col].dropna(), equal_var=False)
    def star(p): return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
    print(f"  {label[:40]:40s}  pre-vs-disr: t={t1:+.2f} p={p1:.3f}{star(p1)}   "
          f"pre-vs-post: t={t2:+.2f} p={p2:.3f}{star(p2)}")

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3: Pre-period correlation matrix
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 3 — PRE-PERIOD CORRELATION MATRIX (key series)")
print("=" * 70)

corr_cols = ["auto_prod","ppi_mv_parts","balt_import_vol","hs8708_cif_B","ppi_petrochem"]
corr_labels = ["AutoProd","PPI_parts","BaltPort","HS8708","PetroChem(placebo)"]
C = PRE[corr_cols].corr()
C.index = corr_labels; C.columns = corr_labels
print(C.round(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Four-panel motivating-evidence plot
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BUILDING FIGURE 1 — Four-panel motivating-evidence plot")
print("=" * 70)

COLLAPSE = pd.Timestamp("2024-03-26")
FULL_OP  = pd.Timestamp("2024-06-10")
XLIM     = (pd.Timestamp("2019-01-01"), pd.Timestamp("2024-12-31"))

fig = plt.figure(figsize=(12, 7.5))
gs = gridspec.GridSpec(2, 2, hspace=0.48, wspace=0.28)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]


# ── Panel A: Baltimore port import volume (shock variable) ─────────────────
ax = axes[0]
s = master["balt_import_vol"] / 1e6
pre_mean = PRE["balt_import_vol"].mean() / 1e6

ax.fill_between(s.index, s.values, 0, alpha=0.10, color=COLOR["baltimore"], zorder=1)
ax.plot(s.index, s.values, color=COLOR["baltimore"], lw=2.0, zorder=2)
ax.axhline(pre_mean, color=INK_SECONDARY, lw=1, ls=":", alpha=0.7, zorder=2)
event_lines(ax, COLLAPSE, FULL_OP, label=False)

ax.scatter(pd.Timestamp("2024-04-01"), s.loc["2024-04-01"], color=COLOR["shock"],
           s=45, zorder=6, edgecolor="white", linewidth=1.2)
ax.annotate("Apr 2024\n−99.5%",
            xy=(pd.Timestamp("2024-04-01"), s.loc["2024-04-01"]),
            xytext=(pd.Timestamp("2022-01-01"), 2.7),
            fontsize=_style.FS_ANNOT, color=COLOR["shock"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLOR["shock"], lw=1.1))

ax.set_title("(A)  Port of Baltimore import volume\nIMF PortWatch Container_Metrics",
             fontsize=_style.FS_TITLE, pad=6)
ax.set_ylabel("Monthly import volume (millions)", fontsize=_style.FS_AXIS)
ax.set_xlim(XLIM); ax.set_ylim(-0.2, 7.5)
ax.text(0.02, 0.05, f"Pre-period mean: {pre_mean:.2f}M",
        transform=ax.transAxes, fontsize=_style.FS_FOOTNOTE, color=INK_SECONDARY)

# ── Panel B: US auto production (primary outcome) ──────────────────────────
ax = axes[1]
s = master["auto_prod"]
pre_mean_p = PRE["auto_prod"].mean()

ax.fill_between(s.index, s.values, 0, alpha=0.10, color=COLOR["auto_prod"], zorder=1)
ax.plot(s.index, s.values, color=COLOR["auto_prod"], lw=2.0, zorder=2)
ax.axhline(pre_mean_p, color=INK_SECONDARY, lw=1, ls=":", alpha=0.7, zorder=2)
event_lines(ax, COLLAPSE, FULL_OP, label=False)

ax.scatter(pd.Timestamp("2024-04-01"), s.loc["2024-04-01"], color=COLOR["shock"],
           s=45, zorder=6, edgecolor="white", linewidth=1.2)
ax.annotate("Apr 2024\n−27.3% YoY",
            xy=(pd.Timestamp("2024-04-01"), s.loc["2024-04-01"]),
            xytext=(pd.Timestamp("2021-06-01"), 55),
            fontsize=_style.FS_ANNOT, color=COLOR["shock"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLOR["shock"], lw=1.1))

ax.set_title("(B)  US domestic auto production (DAUPSA)\nFRED · monthly, seasonally adjusted",
             fontsize=_style.FS_TITLE, pad=6)
ax.set_ylabel("Production (DAUPSA × 1,000 = monthly cars)", fontsize=_style.FS_AXIS)
ax.set_xlim(XLIM)
ax.text(0.02, 0.05, f"Pre-period mean: {pre_mean_p:.1f}k",
        transform=ax.transAxes, fontsize=_style.FS_FOOTNOTE, color=INK_SECONDARY)

# ── Panel C: PPI motor vehicle parts vs placebo ────────────────────────────
ax = axes[2]
ppi = master["ppi_mv_parts"]
plac = master["ppi_petrochem"]
ppi_norm  = ppi  / ppi.iloc[0]  * 100
plac_norm = plac / plac.iloc[0] * 100

ax.plot(ppi_norm.index,  ppi_norm.values,  color=COLOR["ppi_mv_parts"], lw=2.0,
        zorder=3, label="MV parts PPI (outcome)")
ax.plot(plac_norm.index, plac_norm.values, color=COLOR["ppi_petrochem"], lw=1.6,
        ls="--", zorder=2, label="Petrochem PPI (placebo)")
event_lines(ax, COLLAPSE, FULL_OP, label=False)

ax.set_title("(C)  PPI: motor vehicle parts vs placebo\nFRED · Jan 2019 = 100",
             fontsize=_style.FS_TITLE, pad=6)
ax.set_ylabel("Index (Jan 2019 = 100)", fontsize=_style.FS_AXIS)
ax.set_xlim(XLIM)
ax.legend(fontsize=_style.FS_LEGEND, loc="upper left")
ax.annotate("No price spike\n→ logistics cost,\nnot supply scarcity",
            xy=(pd.Timestamp("2024-05-01"), ppi_norm.loc["2024-05-01"]),
            xytext=(pd.Timestamp("2022-05-01"), 148),
            fontsize=_style.FS_FOOTNOTE, color=COLOR["ppi_mv_parts"],
            arrowprops=dict(arrowstyle="->", color=COLOR["ppi_mv_parts"], lw=1))

# ── Panel D: HS 8708 CIF imports — resilience story ─────────────────────────
ax = axes[3]
s = master["hs8708_cif_B"]
pre_mean_ct = PRE["hs8708_cif_B"].mean()

ax.fill_between(s.index, s.values, 0, alpha=0.10, color=COLOR["hs8708"], zorder=1)
ax.plot(s.index, s.values, color=COLOR["hs8708"], lw=2.0, zorder=2)
ax.axhline(pre_mean_ct, color=INK_SECONDARY, lw=1, ls=":", alpha=0.7, zorder=2)
event_lines(ax, COLLAPSE, FULL_OP, label=False)

ax.scatter(pd.Timestamp("2024-04-01"), s.loc["2024-04-01"], color=COLOR["hs8708"],
           s=45, zorder=6, edgecolor="white", linewidth=1.2)
ax.annotate("Apr 2024: +8.2% YoY\n→ imports diverted,\n    not cancelled",
            xy=(pd.Timestamp("2024-04-01"), s.loc["2024-04-01"]),
            xytext=(pd.Timestamp("2020-04-01"), 9.2),
            fontsize=_style.FS_ANNOT, color=COLOR["hs8708"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLOR["hs8708"], lw=1.1))

ax.set_title("(D)  US imports, HS 8708 (motor vehicle parts)\nUN Comtrade · monthly CIF value",
             fontsize=_style.FS_TITLE, pad=6)
ax.set_ylabel("CIF import value (USD billions)", fontsize=_style.FS_AXIS)
ax.set_xlim(XLIM); ax.set_ylim(0, 11)
ax.text(0.02, 0.05, f"Pre-period mean: ${pre_mean_ct:.2f}B/month",
        transform=ax.transAxes, fontsize=_style.FS_FOOTNOTE, color=INK_SECONDARY)

# ── Shared formatting ──────────────────────────────────────────────────────
for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    style_axes(ax)

# ── Shared legend ────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=COLOR["shock"], alpha=0.08, label="Disruption window (Mar 26 – Jun 10, 2024)"),
    plt.Line2D([0], [0], color=COLOR["shock"], lw=1.6, ls="--", label="FSK bridge collapse (Mar 26)"),
    plt.Line2D([0], [0], color=COLOR["recovery"], lw=1.6, ls="--", label="Full channel reopening (Jun 10)"),
    plt.Line2D([0], [0], color=INK_SECONDARY, lw=1, ls=":", label="Pre-period mean"),
]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=_style.FS_LEGEND,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.subplots_adjust(bottom=0.10)
save_figure(fig, "figure1_motivating_evidence", category="main")
