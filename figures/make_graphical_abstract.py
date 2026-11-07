"""Graphical abstract (130 mm x 50 mm, Elsevier aspect ratio 5:13), drawn from the reference outputs.

  python figures/make_graphical_abstract.py

Reads results_reference/TETDC_1.csv, uq_results.json, uq_results_bc.json and terrain_scenarios.json
(or the same files in data/processed/ if a fresh run exists) and writes
figures/output/graphical_abstract.pdf and .png (300 dpi).
"""
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = next(d for d in (os.path.join(ROOT, "data", "processed"), os.path.join(ROOT, "results_reference"))
           if os.path.exists(os.path.join(d, "uq_results_bc.json")))
OUT = os.path.join(ROOT, "figures", "output"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 7})
MM = 1 / 25.4
C = {"summer": "#d35400", "winter": "#2471a3", "shoulder": "#27ae60", "ink": "#333333", "grey": "#7f8c8d",
     "box": "#eef3f9", "edge": "#2c5f8a"}
SEASONS = [("summer", "Summer"), ("winter", "Winter"), ("shoulder", "Spring/\nautumn")]

v = pd.read_csv(os.path.join(SRC, "TETDC_1.csv")).v_ms.to_numpy() * 3.6
u = {t: json.load(open(os.path.join(SRC, f"uq_results{t}.json"))) for t in ("", "_bc")}
topo = u[""]["summer"]["monte_carlo"]["topo_effect"]["mean"]

fig = plt.figure(figsize=(130 * MM, 50 * MM))
bg = fig.add_axes([0, 0, 1, 1], zorder=0); bg.set_axis_off(); bg.set_xlim(0, 1); bg.set_ylim(0, 1)

def box(x, w, title):
    bg.add_patch(FancyBboxPatch((x, 0.05), w, 0.9, boxstyle="round,pad=0.005,rounding_size=0.02",
                                fc=C["box"], ec=C["edge"], lw=0.8))
    bg.text(x + w / 2, 0.88, title, ha="center", va="center", fontsize=7.5, weight="bold", color=C["ink"])

def arrow(x0, x1):
    bg.add_patch(matplotlib.patches.FancyArrow(x0, 0.5, x1 - x0, 0, width=0.008, head_width=0.05,
                 head_length=0.012, length_includes_head=True, color=C["ink"]))

# 1 open data
box(0.01, 0.22, "Open data")
fig.text(0.12, 0.47, "Porto taxi GPS (15 s)\nOpenStreetMap\nCopernicus DEM\nopen 1 Hz GPS\n(bias test)",
         ha="center", va="center", fontsize=7, color=C["ink"], linespacing=1.5)
arrow(0.235, 0.26)

# 2 cycle and offline digital twin
box(0.265, 0.27, "Cycle + offline twin")
ax = fig.add_axes([0.305, 0.36, 0.21, 0.38], zorder=2); ax.patch.set_alpha(0)
ax.plot(v, color=C["edge"], lw=0.5)
ax.set_xlim(0, len(v)); ax.set_ylim(0, max(v) * 1.1)
ax.set_xticks([]); ax.set_yticks([0, 50]); ax.set_yticklabels(["0", "50\nkm/h"], fontsize=6)
ax.tick_params(length=2, pad=1)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title(f"TETDC-1 ({len(v):,} s)", fontsize=7, pad=2)
fig.text(0.40, 0.19, "flat WLTC vs Tehran grade\n3 seasonal auxiliary loads", ha="center", va="center",
         fontsize=6.5, color=C["ink"], linespacing=1.4)
arrow(0.54, 0.565)

# 3 result
box(0.57, 0.42, "Flat WLTC range overestimate")
ax = fig.add_axes([0.665, 0.27, 0.30, 0.50], zorder=2); ax.patch.set_alpha(0)
for i, (s, lab) in enumerate(SEASONS):
    for dy, t, filled in ((0.17, "", False), (-0.17, "_bc", True)):
        m = u[t][s]["monte_carlo"]["range_overestimate_pct"]
        y = len(SEASONS) - 1 - i + dy
        ax.plot([m["p2_5"], m["p97_5"]], [y, y], color=C[s], lw=1.0, alpha=0.9 if filled else 0.5)
        ax.plot(m["mean"], y, "o", ms=3.5, mfc=C[s] if filled else "white", mec=C[s], mew=0.9)
ax.axvline(0, color=C["ink"], lw=0.6)
ax.set_yticks(range(len(SEASONS))); ax.set_yticklabels([lab for _, lab in SEASONS][::-1], fontsize=6.5)
ax.set_xlabel("%  (mean, 95% interval)", fontsize=6.5, labelpad=1)
ax.tick_params(length=2, pad=1)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.plot([], [], "o", ms=3.5, mfc="white", mec=C["ink"], label="original")
ax.plot([], [], "o", ms=3.5, mfc=C["ink"], mec=C["ink"], label="corrected")
ax.set_xlim(-16, 19); ax.set_xticks([-10, 0, 10])
ax.legend(loc="center left", fontsize=6, frameon=False, handletextpad=0.2, borderaxespad=0.1)
fig.text(0.78, 0.11, f"terrain: +{topo:.1f} kWh/100 km at the central grade", ha="center", va="center",
         fontsize=6.5, color=C["ink"])

for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    fig.savefig(os.path.join(OUT, f"graphical_abstract.{ext}"), **kw)
plt.close(fig)
print("written", os.path.join(OUT, "graphical_abstract.pdf"))
