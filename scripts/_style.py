"""
Shared visual design system for every figure in this study.
=============================================================
One palette, one typography scale, one save path convention -- applied
identically across all figures so the final figure set reads as one
coherent, Q1-journal-grade visual system rather than N independently
styled matplotlib defaults.

Palette source: the validated, CVD-safe categorical palette from the
project's `dataviz` design skill (8-hue order chosen to maximize minimum
adjacent perceptual distance under simulated color-vision deficiency;
worst adjacent CVD delta 9.1, worst adjacent normal-vision delta 19.6,
both clear the skill's hard gates). Only light-mode values are used --
these are static print/PDF journal figures, not an interactive UI, so
there is no dark-mode context to design for.

Colors are assigned to semantic ROLES (entities, not chart-local slots)
and reused by identity everywhere that entity appears, per the skill's
"color follows the entity, never its rank" rule. Two entities that never
co-occur in the same figure may share a role (documented inline) rather
than exhausting the palette on distinctions no single figure ever needs
to draw.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

ROOT = Path(__file__).parent.parent
PAPER_ASSETS = ROOT / "paper_assets"
FIG_MAIN = PAPER_ASSETS / "figures" / "main"
FIG_SUPP = PAPER_ASSETS / "figures" / "supplementary"
TAB_MAIN = PAPER_ASSETS / "tables" / "main"
TAB_SUPP = PAPER_ASSETS / "tables" / "supplementary"
for _d in (FIG_MAIN, FIG_SUPP, TAB_MAIN, TAB_SUPP):
    _d.mkdir(parents=True, exist_ok=True)

# ── Core categorical palette (fixed order; never cycled/reassigned) ─────────
BLUE    = "#2a78d6"
GREEN   = "#008300"
MAGENTA = "#e87ba4"
YELLOW  = "#eda100"
AQUA    = "#1baf7a"
ORANGE  = "#eb6834"
VIOLET  = "#4a3aa7"
RED     = "#e34948"

# ── Status colors (reserved; never reused as a categorical series color) ───
GOOD     = "#0ca30c"   # recovery / reopening events
WARNING  = "#fab219"
SERIOUS  = "#ec835a"
CRITICAL = "#d03b3b"   # shock / collapse events

# ── Chrome: surfaces, ink, structural lines ─────────────────────────────────
SURFACE       = "#ffffff"   # pure white -- print/PDF submission, not a UI card
INK_PRIMARY   = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED     = "#898781"   # de-emphasized entities: placebo, excluded series
GRIDLINE      = "#e1e0d9"
BASELINE      = "#c3c2b7"

# ── Semantic entity -> color, fixed across every figure in the study ───────
# Two roles intentionally share a hue where the entities never appear in the
# same figure (Brunswick / ppi_mv_parts; noted below) -- this keeps the
# palette small and memorable rather than exhausting all 8 slots on
# distinctions no single figure ever has to draw.
COLOR = {
    "auto_prod":        VIOLET,    # primary outcome: domestic auto production
    "ppi_mv_parts":      ORANGE,    # price outcome: PPI motor-vehicle parts
    "hs8708":            GREEN,     # trade outcome: HS 8708 imports
    "ppi_petrochem":      INK_MUTED,  # placebo -- de-emphasized, not a peer series
    "baltimore":         BLUE,      # treatment port
    "brunswick":         ORANGE,    # diversion port (shares ORANGE with ppi_mv_parts;
                                     # never co-occurs with it in the same figure)
    "norfolk":           INK_MUTED,   # excluded (noise) -- shown for transparency only
    "port_virginia":      INK_MUTED,  # excluded (noise) -- shown for transparency only
    "shock":             CRITICAL,   # collapse-date event marker (vertical line)
    "recovery":          GOOD,       # reopening-date event marker (vertical line)
    "counterfactual":     INK_SECONDARY,  # modeled/what-if line, dashed
    "significant":       CRITICAL,
    "marginal":          WARNING,
    "not_significant":    INK_MUTED,
}

# ── Typography scale (points) ───────────────────────────────────────────────
FS_TITLE     = 10.5
FS_AXIS      = 9.5
FS_TICK      = 8.5
FS_LEGEND    = 8.5
FS_ANNOT     = 8
FS_FOOTNOTE  = 7.5

SANS = ["Arial", "Helvetica", "Segoe UI", "DejaVu Sans", "sans-serif"]


def apply_style():
    """Call once per script, before building any figure."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   SANS,
        "text.color":        INK_PRIMARY,
        "axes.edgecolor":    BASELINE,
        "axes.labelcolor":   INK_PRIMARY,
        "axes.titlecolor":   INK_PRIMARY,
        "axes.labelsize":    FS_AXIS,
        "axes.titlesize":    FS_TITLE,
        "axes.linewidth":    0.9,
        "xtick.color":       INK_SECONDARY,
        "ytick.color":       INK_SECONDARY,
        "xtick.labelsize":   FS_TICK,
        "ytick.labelsize":   FS_TICK,
        "legend.fontsize":   FS_LEGEND,
        "legend.frameon":    True,
        "legend.framealpha": 0.95,
        "legend.edgecolor":  GRIDLINE,
        "figure.facecolor":  SURFACE,
        "axes.facecolor":    SURFACE,
        "savefig.facecolor": SURFACE,
        "grid.color":        GRIDLINE,
        "grid.linewidth":    0.8,
        "lines.linewidth":   2.0,
        "lines.markersize":  6,
        "patch.linewidth":   0,
    })


def style_axes(ax, grid_axis="y"):
    """Apply consistent spine/grid treatment to one axes."""
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=FS_TICK)


def event_lines(ax, shock_date, recovery_date, shade=True, label=True):
    """Consistent shock/recovery vertical markers + disruption-window shading,
    used identically on every time-series panel across the whole figure set."""
    ax.axvline(shock_date, color=COLOR["shock"], lw=1.6, ls="--",
               alpha=0.9, zorder=4,
               label="FSK collapse (Mar 26)" if label else None)
    ax.axvline(recovery_date, color=COLOR["recovery"], lw=1.6, ls="--",
               alpha=0.9, zorder=4,
               label="Full reopening (Jun 10)" if label else None)
    if shade:
        ax.axvspan(shock_date, recovery_date, color=CRITICAL, alpha=0.06, zorder=0)


def save_figure(fig, stem, category, formats=("pdf", "png")):
    """Save one figure to paper_assets/figures/{main,supplementary}/ in every
    requested format. `stem` excludes extension, e.g. 'figure1_motivating_evidence'.
    PDF is the primary (vector, submission-grade) format; PNG (300 dpi) is a
    high-resolution raster copy for quick viewing / embedding in a working
    manuscript draft."""
    target_dir = FIG_MAIN if category == "main" else FIG_SUPP
    paths = []
    for fmt in formats:
        p = target_dir / f"{stem}.{fmt}"
        dpi = 300 if fmt == "png" else None
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
        paths.append(p)
    plt.close(fig)
    print(f"  Saved {stem}: " + ", ".join(str(p.name) for p in paths) +
          f"  -> {target_dir}")
    return paths
