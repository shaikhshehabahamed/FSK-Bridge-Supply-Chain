"""
Step 8b: Diversion Spillover Visualisation
Figure 5 (main): Four-panel port-network story
  (A) RoRo portcall timeline — Baltimore vs Brunswick, 2019-2021 + 2024
  (B) Market-share slope chart — pre vs post (replaces a pie-chart pair:
      a slope chart states the two shares and their shift precisely,
      where pie-chart angle comparison does not)
  (C) Pre vs post RoRo portcall volume, paired bars
  (D) Import-volume recovery trajectory, Jun-Dec 2024
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import pickle, warnings
warnings.filterwarnings("ignore")

import _style
from _style import COLOR, INK_PRIMARY, INK_SECONDARY, GRIDLINE, save_figure, style_axes
_style.apply_style()

with open(str(DATA_PROC / "diversion_data.pkl"), "rb") as f:
    div = pickle.load(f)

pw          = div["pw"]
pre_totals  = div["pre_totals"]
pre  = pw[pw["date"].dt.year.isin([2019, 2020, 2021])]
post = pw[(pw["date"].dt.year == 2024) & (pw["date"].dt.month >= 7)]
post_totals = div["post_totals"]
pre_shares  = div["pre_shares"]
post_shares = div["post_shares"]
ttest       = div["ttest"]

LABELS = {"baltimore": "Baltimore", "brunswick": "Brunswick (GA)"}
PORTS  = ["baltimore", "brunswick"]
COLOURS = {p: COLOR[p] for p in PORTS}

COLLAPSE = pd.Timestamp("2024-03-26")
FULL_OP  = pd.Timestamp("2024-06-10")

fig = plt.figure(figsize=(12, 8.2))
gs  = gridspec.GridSpec(2, 2, hspace=0.5, wspace=0.32)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]


# ── Panel A: RoRo portcall timeline ─────────────────────────────────────────
ax = axes[0]
_full_months = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
for port in PORTS:
    s = pw[pw["port_label"] == port].set_index("date").sort_index()
    # Reindex onto every month 2019-2024 (not just the months with data) so
    # matplotlib draws a genuine gap -- rather than a straight connecting
    # line implying interpolated trend -- across the 2021-09 to 2024-06
    # PortWatch coverage gap (Step 8A1/8A2 in 08_diversion_spillover.py).
    s = s.reindex(_full_months)
    ax.plot(s.index, s["portcalls_roro"], color=COLOURS[port], lw=1.8,
            marker="o", markersize=3, label=LABELS[port], zorder=3)
    if port == "baltimore":
        ax.fill_between(s.index, s["portcalls_roro"], 0, alpha=0.08,
                        color=COLOURS[port], zorder=1)

ax.axvspan(pd.Timestamp("2021-09-01"), pd.Timestamp("2024-06-01"),
           color=GRIDLINE, alpha=0.5, zorder=0, hatch="//", edgecolor="white")
ax.text(pd.Timestamp("2022-09-01"), 58, "PortWatch data\ngap (2021–2024)",
        fontsize=_style.FS_FOOTNOTE, color=INK_SECONDARY, ha="center")

ax.axvline(COLLAPSE, color=COLOR["shock"], lw=1.5, ls="--", alpha=0.9, zorder=4)
ax.axvline(FULL_OP,  color=COLOR["recovery"], lw=1.5, ls="--", alpha=0.9, zorder=4)
ax.axvspan(COLLAPSE, FULL_OP, color=COLOR["shock"], alpha=0.06, zorder=1)

brun_post_m = pw[(pw["port_label"] == "brunswick") & (pw["date"].dt.year == 2024) &
                 (pw["date"].dt.month >= 7)]["portcalls_roro"].mean()
brun_d = ttest["brunswick"]["d"]
ax.annotate(f"Brunswick +77% YoY\n(p<0.001, d={brun_d:.2f})",
            xy=(pd.Timestamp("2024-09-01"), brun_post_m),
            xytext=(pd.Timestamp("2023-02-01"), 70),
            fontsize=_style.FS_ANNOT, color=COLOURS["brunswick"], fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=COLOURS["brunswick"], lw=1.1))

ax.set_title("(A)  Monthly RoRo port calls: Baltimore vs. Brunswick", fontsize=_style.FS_TITLE, pad=6)
ax.set_ylabel("Monthly RoRo port calls", fontsize=_style.FS_AXIS)
ax.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2024-12-31"))
ax.set_ylim(0, 82)
ax.legend(fontsize=_style.FS_LEGEND, loc="upper left")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
style_axes(ax)

# ── Panel B: Market-share slope chart (replaces pie-chart pair) ────────────
ax = axes[1]
xs = [0, 1]
for port in PORTS:
    ys = [pre_shares[port] * 100, post_shares[port] * 100]
    ax.plot(xs, ys, color=COLOURS[port], lw=2.4, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=1.3, zorder=3)
    delta = ys[1] - ys[0]
    ax.annotate(f"{LABELS[port]}\n{ys[0]:.1f}% → {ys[1]:.1f}%  ({delta:+.1f} pp)",
                xy=(1, ys[1]), xytext=(1.08, ys[1]),
                fontsize=_style.FS_ANNOT, color=COLOURS[port], fontweight="bold",
                va="center", ha="left")

ax.set_xlim(-0.15, 1.75)
ax.set_ylim(25, 70)
ax.set_xticks(xs)
ax.set_xticklabels(["PRE\n(2019–2021)", "POST\n(Jul–Dec 2024)"], fontsize=_style.FS_AXIS)
ax.set_ylabel("RoRo portcall market share (%)", fontsize=_style.FS_AXIS)
ax.set_title("(B)  RoRo market share: pre vs. post", fontsize=_style.FS_TITLE, pad=6)
style_axes(ax, grid_axis="y")
ax.spines[["bottom"]].set_visible(True)

# ── Panel C: Pre vs post RoRo portcall volume, paired bars ──────────────────
ax = axes[2]
x = np.arange(len(PORTS))
pre_v  = [pre_totals[p] for p in PORTS]
post_v = [post_totals[p] for p in PORTS]
delta  = [post_v[i] - pre_v[i] for i in range(len(PORTS))]

bar_w = 0.32
ax.bar(x - bar_w/2 - 0.02, pre_v, bar_w, color=[COLOURS[p] for p in PORTS],
       alpha=0.45, label="Pre-period (2019–2021 avg)", edgecolor="white", lw=1.0)
ax.bar(x + bar_w/2 + 0.02, post_v, bar_w, color=[COLOURS[p] for p in PORTS],
       alpha=0.95, label="Post-period (Jul–Dec 2024 avg)", edgecolor="white", lw=1.0)

for i, (dv, pv, pst) in enumerate(zip(delta, pre_v, post_v)):
    ax.annotate(f"{dv:+.1f} ({dv/pv*100:+.0f}%)",
                xy=(i + bar_w/2 + 0.02, pst + 1.5), fontsize=_style.FS_FOOTNOTE, ha="center",
                fontweight="bold", color=COLOURS[PORTS[i]])
    t_res = ttest[PORTS[i]]
    sig = "***" if t_res["p"] < 0.001 else "**" if t_res["p"] < 0.01 else \
          "*" if t_res["p"] < 0.05 else "ns"
    ax.text(i, 65, f"p={t_res['p']:.4f} {sig}", ha="center",
            fontsize=_style.FS_FOOTNOTE, color=COLOURS[PORTS[i]], fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([LABELS[p] for p in PORTS], fontsize=_style.FS_AXIS)
ax.set_ylabel("Avg. monthly RoRo port calls", fontsize=_style.FS_AXIS)
ax.set_title("(C)  Pre vs. post RoRo portcall volume", fontsize=_style.FS_TITLE, pad=6)
ax.legend(fontsize=_style.FS_FOOTNOTE, loc="upper left")
ax.set_ylim(0, 70)
style_axes(ax)

# ── Panel D: Recovery trajectory — import_roro ──────────────────────────────
ax = axes[3]
months_2024 = pw[(pw["date"].dt.year == 2024) & (pw["date"].dt.month >= 6)]
for port in PORTS:
    s = months_2024[months_2024["port_label"] == port].sort_values("date")
    ax.plot(s["date"], s["import_roro"], color=COLOURS[port], lw=1.8,
            marker="o", markersize=5, label=LABELS[port], zorder=3)
    pre_m = pre[pre["port_label"] == port]["import_roro"].mean()
    ax.axhline(pre_m, color=COLOURS[port], lw=1.1, ls=":", alpha=0.6)

ax.axvline(FULL_OP, color=COLOR["recovery"], lw=1.4, ls="--", alpha=0.9, zorder=4,
           label="Full reopening")

ax.set_title("(D)  RoRo import-volume recovery trajectory\nJun–Dec 2024 vs. pre-period averages (dotted)",
             fontsize=_style.FS_TITLE, pad=6)
ax.set_ylabel("Monthly RoRo import volume", fontsize=_style.FS_AXIS)
ax.legend(fontsize=_style.FS_FOOTNOTE, loc="lower right", ncol=1)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 8, 10, 12]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.text(0.03, 0.94, "Brunswick > Baltimore\nby Jul 2024",
        transform=ax.transAxes, fontsize=_style.FS_ANNOT, color=INK_SECONDARY,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GRIDLINE))
style_axes(ax)

plt.tight_layout(rect=[0, 0, 1, 1])
save_figure(fig, "figure5_diversion", category="main")
print("Step 8b complete")
