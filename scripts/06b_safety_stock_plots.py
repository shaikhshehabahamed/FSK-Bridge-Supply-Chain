"""
Step 6b: Safety Stock Visualisation
Figure 4 (main): Three-panel safety stock story
  (A) theta sensitivity — implied SS days across Baltimore supply share
  (B) Production recovery timeline with buffer zones
  (C) Policy chart — required SS for target production maintenance
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import _style
from _style import COLOR, GOOD, WARNING, CRITICAL, INK_PRIMARY, INK_SECONDARY, save_figure, style_axes
_style.apply_style()

master  = pd.read_csv(str(DATA_PROC / "master.csv"),
                       parse_dates=["date"], index_col="date")
df_ss   = pd.read_csv(str(DATA_PROC / "safety_stock_results.csv"))
summary = pd.read_csv(str(DATA_PROC / "ss_summary.csv"),
                       index_col=0, header=None).squeeze()

theta_c  = float(summary["theta_central"])
theta_lo = float(summary["theta_low"])
theta_hi = float(summary["theta_high"])
ss_c     = float(summary["ss_central_days"])
ss_lo    = float(summary["ss_range_low_days"])
ss_hi    = float(summary["ss_range_high_days"])
apr_drop = float(summary["apr_drop_pct"])
D        = 30   # closure days

MAIN = COLOR["baltimore"]
SHADE = "#eaf1fb"

fig = plt.figure(figsize=(13, 4.6))
gs = gridspec.GridSpec(1, 3, wspace=0.42)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]


# ── Panel A: theta sensitivity ──────────────────────────────────────────────
ax = axes[0]
valid   = df_ss[df_ss["valid"]]
invalid = df_ss[~df_ss["valid"]]

ax.fill_between(valid["theta"], ss_lo, ss_hi, alpha=0.12, color=MAIN, zorder=1)
ax.plot(valid["theta"], valid["ss_days"], color=MAIN, lw=2.2, zorder=3)
ax.scatter(valid["theta"], valid["ss_days"], color=MAIN, s=28, zorder=4,
           edgecolor="white", linewidth=0.8)
if len(invalid):
    ax.scatter(invalid["theta"], [0]*len(invalid), color=CRITICAL, s=40,
               marker="x", zorder=4, label="Infeasible (θ too small)")

ax.axvline(theta_c, color=MAIN, lw=1.3, ls="--", alpha=0.8, zorder=2)
ax.axvspan(theta_lo, theta_hi, color=SHADE, alpha=0.9, zorder=0,
           label=f"θ uncertainty band [{theta_lo:.2f}–{theta_hi:.2f}]")
ax.scatter([theta_c], [ss_c], color=MAIN, s=95, zorder=5, edgecolor="white",
           linewidth=1.2, label=f"Central: {ss_c:.1f} days")

ax.axhline(6,  color=GOOD, lw=1.1, ls=":", alpha=0.8, label="Toyota JIT target (5–7d)")
ax.axhline(17, color=WARNING, lw=1.1, ls="--", alpha=0.8, label="Post-pandemic target (14–21d)")

ax.set_xlabel("Baltimore's automotive supply share (θ)", fontsize=_style.FS_AXIS)
ax.set_ylabel("Implied safety stock (days-of-supply)", fontsize=_style.FS_AXIS)
ax.set_title("(A)  Sensitivity: implied SS vs. θ\nApril 2024, YoY production drop = 27.3%",
             fontsize=_style.FS_TITLE, pad=6)
ax.legend(fontsize=_style.FS_FOOTNOTE, loc="upper left")
ax.set_xlim(0.15, 0.85); ax.set_ylim(-5, 30)
ax.axhline(0, color=INK_SECONDARY, lw=0.7, alpha=0.5)
style_axes(ax)

# ── Panel B: Recovery timeline ──────────────────────────────────────────────
ax = axes[1]
months = ["2024-03-01","2024-04-01","2024-05-01","2024-06-01",
          "2024-07-01","2024-08-01","2024-09-01"]
prod = master["auto_prod"]

actual = [prod.loc[m] for m in months]
baseline = [prod.loc[m.replace("2024","2023")] for m in months]
yoy_pct = [(a/b - 1)*100 for a, b in zip(actual, baseline)]
labels  = ["Mar","Apr","May","Jun","Jul","Aug","Sep"]

x = np.arange(len(months))
bar_colors = [CRITICAL if v < -20 else WARNING if v < -10 else MAIN for v in yoy_pct]
ax.bar(x, yoy_pct, color=bar_colors, alpha=0.88, zorder=2, edgecolor="white", lw=1.0, width=0.62)

ax.axhline(0, color=INK_PRIMARY, lw=1.0, alpha=0.5)
ax.axhline(-apr_drop*100, color=CRITICAL, lw=1.2, ls="--", alpha=0.7,
           label=f"April peak: −{apr_drop*100:.1f}%")

ax.axvspan(-0.5, 1.5, color=CRITICAL, alpha=0.05, zorder=0, label="Full closure")
ax.axvspan(1.5,  2.5, color=WARNING, alpha=0.08, zorder=0, label="Partial channel")
ax.axvspan(2.5,  6.5, color=GOOD, alpha=0.05, zorder=0, label="Phased recovery")

for i, v in enumerate(yoy_pct):
    # Label sits just INSIDE the bar's tip (toward zero), not beyond it --
    # placing it past the tip drops it onto the light background band
    # behind the bar, where white text has no contrast.
    ax.text(i, v + 1.6, f"{v:.1f}%", ha="center", va="bottom", fontsize=_style.FS_FOOTNOTE,
            fontweight="bold", color="white")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=_style.FS_TICK)
ax.set_ylabel("YoY production change (%)", fontsize=_style.FS_AXIS)
ax.set_title("(B)  Recovery timeline: production YoY\nDAUPSA vs. same month 2023",
             fontsize=_style.FS_TITLE, pad=6)
ax.legend(fontsize=_style.FS_FOOTNOTE, loc="lower right")
ax.set_ylim(-35, 5)
style_axes(ax)

# ── Panel C: Policy chart — required SS for production targets ──────────────
ax = axes[2]
theta_range = np.linspace(0.30, 0.85, 100)
targets = [(0.00, "Zero decline (full continuity)", CRITICAL, "-"),
           (0.05, "≤ 5% decline",                    WARNING,  "--"),
           (0.10, "≤ 10% decline",                   GOOD,     ":")]

for target, label, col, ls in targets:
    required_ss = (1.0 - target / theta_range) * D
    ax.plot(theta_range, required_ss, color=col, lw=1.9, ls=ls, label=label)

ax.fill_betweenx([0, 35], theta_lo, theta_hi, color=SHADE, alpha=0.9, zorder=0)
ax.axvline(theta_c, color=MAIN, lw=1.3, ls="--", alpha=0.7)
ax.scatter([theta_c], [ss_c], color=MAIN, s=85, zorder=5, edgecolor="white",
           linewidth=1.2, label=f"Implied actual: {ss_c:.1f}d @ θ={theta_c:.2f}")
ax.axhline(ss_c, color=MAIN, lw=0.8, ls=":", alpha=0.6)

ax.axhspan(5,  7,  color=GOOD, alpha=0.08, zorder=0)
ax.axhspan(14, 21, color=WARNING, alpha=0.08, zorder=0)
ax.text(0.315, 6,   "Toyota JIT", fontsize=_style.FS_FOOTNOTE, color=GOOD, va="center")
ax.text(0.315, 17.5, "Post-pandemic target", fontsize=_style.FS_FOOTNOTE, color=WARNING, va="center")

ax.set_xlabel("Baltimore's automotive supply share (θ)", fontsize=_style.FS_AXIS)
ax.set_ylabel("Required safety stock (days-of-supply)", fontsize=_style.FS_AXIS)
ax.set_title("(C)  Policy: required SS to limit decline\nClosure duration D = 30 days",
             fontsize=_style.FS_TITLE, pad=6)
ax.set_xlim(0.28, 0.85); ax.set_ylim(0, 35)
ax.legend(fontsize=_style.FS_FOOTNOTE, loc="upper right")
style_axes(ax)

plt.tight_layout(rect=[0, 0, 1, 1])
save_figure(fig, "figure4_safety_stock", category="main")
print("Step 6b complete")
