"""
Step 3b: UN Comtrade Validation Plot
Shows: US HS 8708 imports (CIF value) 2019-2024.
Key finding: NO trade-volume shock -> cargo was diverted, not cancelled.
The production disruption (DAUPSA) is the COST of rerouting, not a supply shortage.
Figure S6 (supplementary).
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np

import _style
from _style import COLOR, INK_SECONDARY, save_figure, style_axes, event_lines
_style.apply_style()

df = pd.read_csv(str(DATA_RAW / "comtrade_hs8708_raw.csv"),
                 parse_dates=["date"], index_col="date")
fred = pd.read_csv(str(DATA_RAW / "fred_raw.csv"),
                   parse_dates=["date"], index_col="date")

COLLAPSE = pd.Timestamp("2024-03-26")
FULL_OP  = pd.Timestamp("2024-06-10")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

# ── Panel A: Total HS 8708 imports — NO shock in trade value ────────────────
ax = axes[0]
s = df["cif_usd"] / 1e9
# Which months were originally missing from the API (before interpolation)
# is read from the "interpolated" column 03_comtrade_data.py saves alongside
# the data, rather than a hardcoded date list -- so this marker stays correct
# if the raw data is ever re-collected and the set of missing periods shifts.
interp_dates = df.index[df["interpolated"]]

ax.fill_between(s.index, s.values, 0, alpha=0.10, color=COLOR["hs8708"], zorder=1)
ax.plot(s.index, s.values, color=COLOR["hs8708"], lw=1.6, zorder=2, label="CIF import value")
ax.scatter(interp_dates, s.reindex(interp_dates).values, color="white",
           edgecolors=COLOR["hs8708"], s=26, zorder=4, label="Interpolated", linewidth=1.1)
event_lines(ax, COLLAPSE, FULL_OP, label=False)

pre_mean = df.loc["2019-01":"2023-12", "cif_usd"].mean() / 1e9
ax.axhline(pre_mean, color=INK_SECONDARY, lw=1, ls=":", alpha=0.8)
ax.text(pd.Timestamp("2019-03-01"), pre_mean + 0.15, f"Pre-period mean: ${pre_mean:.2f}B",
        fontsize=_style.FS_FOOTNOTE, color=INK_SECONDARY)

apr24 = s.loc["2024-04-01"]
ax.annotate(f"Apr 2024: ${apr24:.2f}B (+8.2% YoY)\n→ no supply collapse",
            xy=(pd.Timestamp("2024-04-01"), apr24),
            xytext=(pd.Timestamp("2021-04-01"), 2.3),
            fontsize=_style.FS_ANNOT, color=COLOR["hs8708"], fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=COLOR["hs8708"], lw=1.1))

ax.set_title("(A)  US imports, HS 8708 (motor vehicle parts)\nUN Comtrade · monthly CIF value",
             fontsize=_style.FS_TITLE, pad=6)
ax.set_ylabel("CIF import value (USD billions)", fontsize=_style.FS_AXIS)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2024-12-31"))
ax.set_ylim(0, 10.5)
ax.legend(fontsize=_style.FS_FOOTNOTE, loc="upper left")
style_axes(ax)

# ── Panel B: DAUPSA production — shock IS visible ────────────────────────────
ax2 = axes[1]
prod = fred["auto_production"].loc["2019-01-01":"2024-12-01"]
ax2.plot(prod.index, prod.values, color=COLOR["auto_prod"], lw=1.6, zorder=2)
ax2.fill_between(prod.index, prod.values, 0, alpha=0.10, color=COLOR["auto_prod"], zorder=1)
event_lines(ax2, COLLAPSE, FULL_OP, label=True)

apr_prod = prod.loc["2024-04-01"]; mar_prod = prod.loc["2024-03-01"]
ax2.scatter(pd.Timestamp("2024-04-01"), apr_prod, color=COLOR["shock"], s=40, zorder=5,
            edgecolor="white", linewidth=1.0)
ax2.annotate(f"Apr 2024: {apr_prod:.1f}k ({(1-apr_prod/mar_prod)*100:.0f}% from Mar)\n→ JIT production hit",
             xy=(pd.Timestamp("2024-04-01"), apr_prod),
             xytext=(pd.Timestamp("2020-01-01"), 90),
             fontsize=_style.FS_ANNOT, color=COLOR["shock"], fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=COLOR["shock"], lw=1.1))

ax2.set_title("(B)  US domestic auto production (DAUPSA)\nFRED · monthly, seasonally adjusted",
              fontsize=_style.FS_TITLE, pad=6)
ax2.set_ylabel("Production (DAUPSA × 1,000 = monthly cars)", fontsize=_style.FS_AXIS)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2024-12-31"))
ax2.legend(fontsize=_style.FS_FOOTNOTE, loc="upper left")
style_axes(ax2)


plt.tight_layout(rect=[0, 0, 1, 1])
save_figure(fig, "figureS6_comtrade_validation", category="supplementary")
