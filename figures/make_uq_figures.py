"""Figures that depend only on the uncertainty runs and the supplementary bias test.

  python figures/make_uq_figures.py

Fig. 6: Monte-Carlo distributions of the range overestimate, original and with the reconstruction-bias correction
        (reads data/processed/uq_results*.json and data/processed/uncertainty/mc*_Y*.npy).
Fig. 8: reconstruction bias of 15 s GPS data (reads results/supplement/ or, if absent, results_reference/supplement/).
"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "data", "processed")
SUP = next(d for d in [os.path.join(ROOT, "results", "supplement"), os.path.join(ROOT, "results_reference", "supplement")]
           if os.path.exists(os.path.join(d, "interp_control_per_trip.csv")))
OUT = os.path.join(ROOT, "figures", "output"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42})
MM = 1 / 25.4
SINGLE, DOUBLE = 90 * MM, 190 * MM
C = {"summer": "#d35400", "winter": "#2471a3", "shoulder": "#27ae60", "ink": "#333333", "grey": "#7f8c8d"}
SEASONS = [("summer", "Summer"), ("winter", "Winter"), ("shoulder", "Spring/autumn")]

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# Fig. 6 Monte-Carlo distributions: original vs bias-corrected -----------------------------
u = {t: json.load(open(os.path.join(P, f"uq_results{t}.json"))) for t in ("", "_bc")}
fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 58 * MM), sharey=True)
bins = np.arange(-20, 30, 0.5)
for ax, (s, lab) in zip(axs, SEASONS):
    for t, style in (("", "orig"), ("_bc", "bc")):
        Y = np.vstack([np.load(f) for f in sorted(glob.glob(os.path.join(P, "uncertainty", f"mc{t}_{s}_Y*.npy")))])[:, 8]
        m = u[t][s]["monte_carlo"]["range_overestimate_pct"]
        txt = f"{'Original' if t == '' else 'Corrected'}: {m['mean']:.1f}% ({m['prob_positive']*100:.0f}% >0)"
        if style == "orig":
            ax.hist(Y, bins=bins, histtype="step", color=C["grey"], lw=0.9, label=txt)
        else:
            ax.hist(Y, bins=bins, histtype="stepfilled", alpha=0.35, color=C[s], label=txt)
            ax.hist(Y, bins=bins, histtype="step", color=C[s], lw=0.8)
    # ensemble of 11 construction runs, bias-corrected, scaled to 10,000 draws for comparison
    ens_files = sorted(glob.glob(os.path.join(P, "ensemble", "seed*", f"mc_bc_{s}_Y.npy")))
    if ens_files and os.path.exists(os.path.join(P, "uq_ensemble.json")):
        Ye = np.vstack([np.load(f) for f in ens_files])[:, 8]
        me = json.load(open(os.path.join(P, "uq_ensemble.json")))["seasons"][s]["mc_bc"]["pooled_all"]["range_overestimate_pct"]
        ax.hist(Ye, bins=bins, weights=np.full(len(Ye), 10000 / len(Ye)), histtype="step", color="k", lw=0.9, ls="--",
                label=f"Corr., 11 cycles: {me['mean']:.1f}% ({me['share_positive']*100:.0f}% >0)")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_title(lab, color=C["ink"]); ax.grid(alpha=0.3, lw=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, loc="upper left", fontsize=6.5, handlelength=1.3, borderaxespad=0.3)
    ax.set_xlabel("WLTC range overestimate (%)")
axs[0].set_ylabel("Monte-Carlo runs")
for ax in axs: ax.set_ylim(0, ax.get_ylim()[1] * 1.6); ax.set_xlim(-20, 30)
save(fig, "Fig6_MC_overestimate")

# Fig. 8 Reconstruction bias ------------------------------------------------------------------
b = pd.read_csv(os.path.join(SUP, "interp_bias_per_trip.csv")); b = b[b.used == True]
c = pd.read_csv(os.path.join(SUP, "interp_control_per_trip.csv")).dropna()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 66 * MM))
x, y = b.kwh100_summer_true, b.kwh100_summer_rec
a1.scatter(x, y, s=14, color=C["summer"], edgecolor="white", linewidth=0.5, zorder=3)
lim = [min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
a1.plot(lim, lim, color=C["ink"], lw=0.7, ls="--", label="1:1")
a1.set_xlim(lim); a1.set_ylim(lim); a1.set_aspect("equal")
a1.set_xlabel("1 Hz GPS speed (kWh/100 km)"); a1.set_ylabel("Rebuilt from 15 s positions (kWh/100 km)")
a1.set_title("(a) 54 trips, summer auxiliary load", loc="left"); a1.grid(alpha=0.3, lw=0.4); a1.legend(frameon=False, loc="upper left")
steps = [1, 5, 10, 15]
series = [("kwh", "Energy use", C["summer"], "o"), ("a_rms", "RMS acceleration", C["winter"], "s"), ("P_idle", "Idle share", C["shoulder"], "^")]
for key, lab, col, mk in series:
    med = [float(((c[f"{key}_pos{st}"] - c[f"{key}_pos1"]) / c[f"{key}_pos1"] * 100).median()) for st in steps]
    a2.plot(steps, med, color=col, lw=1.6, marker=mk, ms=4.5, label=lab)
    a2.annotate(f"{med[-1]:.0f}%", (steps[-1], med[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5, color=C["ink"])
a2.axhline(0, color="k", lw=0.6)
a2.set_xticks(steps); a2.set_xlim(0, 17.5)
a2.set_xlabel("GPS sampling interval used for reconstruction (s)"); a2.set_ylabel("Median change vs 1 s reconstruction (%)")
a2.set_title("(b) Effect of the sampling interval", loc="left"); a2.grid(alpha=0.3, lw=0.4); a2.legend(frameon=False, loc="lower left")
save(fig, "Fig8_reconstruction_bias")
print("figures written to", OUT)
