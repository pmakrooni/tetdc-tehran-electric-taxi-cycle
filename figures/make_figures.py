"""Generate all article figures from the verified pipeline outputs.

  python figures/make_figures.py

Reads data/processed/ (produced by run_all.py) and writes figures/output/Fig*.pdf (vector) and .png.
"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from wltp.cycles import class3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "figures", "output"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42})
MM = 1 / 25.4
SINGLE, DOUBLE = 90 * MM, 190 * MM
C = {"TETDC": "#1f5fa8", "WLTC": "#c0392b", "NEDC": "#7f8c8d", "summer": "#d35400", "winter": "#2471a3", "shoulder": "#27ae60"}
SEASONS = [("summer", "Summer"), ("winter", "Winter"), ("shoulder", "Spring/autumn")]

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

vT = pd.read_csv(os.path.join(P, "TETDC_1.csv"))["v_ms"].to_numpy()
vW = np.array(class3.class_data_b()["cycle"], float) / 3.6

# Fig. 1 workflow ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(DOUBLE, 58 * MM)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
boxes = [(0.5, "Open data", "Porto taxi GPS\n(1.71 M trips, 15 s)\nOpenStreetMap\nCopernicus GLO-30\nofficial WLTC trace"),
         (20.5, "Trips and\nmicro-trips", "6-rule cleaning\n1.59 M trips kept\n3% sample: 200,365\nmicro-trips\n(PCA + K-means:\ndescriptive)"),
         (40.5, "Driving cycle", "recombination of\n37,542 micro-trips\nfrom 9,000 traces\nTETDC-1 (1,920 s)"),
         (60.5, "Offline digital\ntwin", "longitudinal\nvehicle model\nTETDC-1 vs WLTC\nflat vs Tehran grade\n3 seasonal loads"),
         (80.5, "Checks and\nuncertainty", "1 Hz bias test\n19 Tehran cycles\n10,000 paired\nMonte-Carlo draws\n11 construction runs\nSobol' indices")]
for x, t, s_ in boxes:
    ax.add_patch(FancyBboxPatch((x, 1.5), 18, 27, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#eef3fa", ec="#1f5fa8", lw=0.8))
    ax.text(x + 9, 24.5, t, ha="center", va="center", fontsize=7.2, weight="bold", linespacing=1.15)
    ax.text(x + 9, 11.5, s_, ha="center", va="center", fontsize=6.4, linespacing=1.35)
for x in (18.8, 38.8, 58.8, 78.8):
    ax.annotate("", xy=(x + 1.6, 15), xytext=(x, 15), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333"))
save(fig, "Fig1_workflow")

# Fig. 2 speed profiles -----------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(DOUBLE, 80 * MM), sharex=True, sharey=True)
for ax, v, name in ((axs[0], vT, "TETDC-1"), (axs[1], vW, "WLTC Class 3b (official)")):
    key = "TETDC" if name.startswith("TETDC") else "WLTC"
    ax.plot(np.arange(len(v)) / 60, v * 3.6, lw=0.7, color=C[key]); ax.fill_between(np.arange(len(v)) / 60, v * 3.6, color=C[key], alpha=0.15, lw=0)
    ax.set_ylabel("Speed (km/h)"); ax.grid(alpha=0.3, lw=0.4)
    ax.text(0.01, 0.92, f"{name}: {len(v)} s, {np.trapezoid(v)/1000:.2f} km, mean {v.mean()*3.6:.1f} km/h, max {v.max()*3.6:.1f} km/h",
            transform=ax.transAxes, fontsize=7, va="top")
axs[1].set_xlabel("Time (min)"); axs[0].set_ylim(0, 140)
save(fig, "Fig2_speed_profiles")

# Fig. 3 speed-acceleration frequency distributions ---------------------------
fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 70 * MM), sharey=True)
for ax, v, name in ((axs[0], vT, "TETDC-1"), (axs[1], vW, "WLTC Class 3b")):
    a = np.gradient(v); m = v > 0.5
    h = ax.hist2d(v[m] * 3.6, a[m], bins=[np.arange(0, 141, 5), np.arange(-2, 2.01, 0.2)], cmap="Blues", density=True, cmin=1e-9)
    ax.set_title(name); ax.set_xlabel("Speed (km/h)")
axs[0].set_ylabel("Acceleration (m/s$^2$)")
cb = fig.colorbar(h[3], ax=axs, fraction=0.025, pad=0.02); cb.set_label("Relative frequency")
save(fig, "Fig3_SAFD")

# Fig. 4 terrain ---------------------------------------------------------------
e = pd.read_csv(os.path.join(P, "edges_with_grade.csv"), usecols=["length_m", "grade_pct"]).dropna()
fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 62 * MM), gridspec_kw={"width_ratios": [1, 1.3]})
g = e["grade_pct"].abs().to_numpy(); L = e["length_m"].to_numpy() / 1000
axs[0].hist(g[g <= 15], bins=np.arange(0, 15.01, 0.5), weights=L[g <= 15], color="#7d6608", alpha=0.8)
axs[0].text(14.8, axs[0].get_ylim()[1] * 0.6, f"{L[g > 15].sum() / L.sum() * 100:.1f}% of length\n> 15% (not shown)", ha="right", fontsize=6.3)
axs[0].set_xlabel("|Road grade| (%), from surface model"); axs[0].set_ylabel("Network length (km)")
axs[0].axvline(1.86, color="k", ls="--", lw=0.7); axs[0].axvline(3.28, color="k", ls=":", lw=0.8)
axs[0].text(2.0, axs[0].get_ylim()[1] * 0.92, "1.86%", fontsize=6.5); axs[0].text(3.45, axs[0].get_ylim()[1] * 0.80, "3.28%", fontsize=6.5)
axs[0].set_title("(a) Grade distribution, 21,049 km network")
prof = [f for f in glob.glob(os.path.join(P, "corridor_profiles", "*.csv")) if "ولی_عصر_خیابان" in os.path.basename(f)][0]
d = pd.read_csv(prof)
axs[1].plot(d["s_m"] / 1000, d["elev_m"], color="#7d6608", lw=1)
axs[1].set_xlabel("Distance along corridor (km)"); axs[1].set_ylabel("Elevation (m)")
axs[1].set_title(f"(b) Valiasr Street: {d['elev_m'].max()-d['elev_m'].min():.0f} m over {d['s_m'].max()/1000:.1f} km"); axs[1].grid(alpha=0.3, lw=0.4)
fig.tight_layout(); save(fig, "Fig4_terrain")

# Fig. 5 seasonal energy (nominal vehicle; Tehran grade at the central value, whiskers = grade bounds) ----
b = json.load(open(os.path.join(P, "baseline_comparison.json")))["seasons"]
ts = json.load(open(os.path.join(P, "terrain_scenarios.json")))["seasons"]
fig, ax = plt.subplots(figsize=(SINGLE * 1.25, 62 * MM))
x = np.arange(3); w = 0.26
series = (("WLTC, flat", C["WLTC"], [b[s]["WLTC_flat"]["kWh_per_100km"] for s, _ in SEASONS], None),
          ("TETDC-1, flat", "#7fa7d9", [ts[s]["flat"] for s, _ in SEASONS], None),
          ("TETDC-1, Tehran grade (2.57%; bars: 1.86-3.28%)", C["TETDC"], [ts[s]["roundtrip_central"] for s, _ in SEASONS],
           [[ts[s]["roundtrip_central"] - ts[s]["roundtrip_lower"] for s, _ in SEASONS],
            [ts[s]["roundtrip_upper"] - ts[s]["roundtrip_central"] for s, _ in SEASONS]]))
for i, (lab, col, vals, err) in enumerate(series):
    ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col, yerr=err, error_kw=dict(lw=0.6, capsize=2) if err else None)
    for k, (xi, v) in enumerate(zip(x + (i - 1) * w, vals)):
        top = v + (err[1][k] if err else 0)
        ax.text(xi, top + 0.3, f"{v:.1f}", ha="center", fontsize=6)
ax.set_xticks(x, [l for _, l in SEASONS]); ax.set_ylabel("Energy use (kWh/100 km)"); ax.set_ylim(0, 28)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2); ax.grid(axis="y", alpha=0.3, lw=0.4)
save(fig, "Fig5_seasonal_energy")

# Fig. 6 and Fig. 8 are produced by figures/make_uq_figures.py
u = json.load(open(os.path.join(P, "uq_results.json")))

# Fig. 7 Sobol total-order indices: original and bias-corrected analysis ----------------------------
ubc = json.load(open(os.path.join(P, "uq_results_bc.json")))
names = {"recon_bias": "Reconstruction bias", "P_aux_W": "Auxiliary load", "grade_frac": "Road grade", "Cd": "Drag coefficient",
         "Cr": "Rolling resistance", "A_m2": "Frontal area", "rho": "Air density", "eta_regen": "Regen. efficiency"}
fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 66 * MM), sharey=True)
keys = list(names); y = np.arange(len(keys)); h = 0.26
for ax, res, title in ((axs[0], u, "(a) Original analysis"), (axs[1], ubc, "(b) With reconstruction-bias correction")):
    for i, (s, lab) in enumerate(SEASONS):
        st = res[s]["sobol"]["range_overestimate_pct"]
        vals = [st[k]["ST"] if k in st else 0.0 for k in keys]; errs = [st[k]["ST_ci"] if k in st else 0.0 for k in keys]
        ax.barh(y + (i - 1) * h, vals, h, xerr=errs, color=C[s], label=lab, error_kw=dict(lw=0.5, capsize=1.5))
    ax.set_title(title, loc="left"); ax.set_xlabel("Sobol' total-order index $S_T$"); ax.grid(axis="x", alpha=0.3, lw=0.4)
    ax.set_xlim(0, 0.8)
axs[0].set_yticks(y, [names[k] for k in keys]); axs[0].invert_yaxis()
axs[1].legend(frameon=False, loc="lower right")
fig.tight_layout(); save(fig, "Fig7_Sobol")
print("figures written to", OUT)
