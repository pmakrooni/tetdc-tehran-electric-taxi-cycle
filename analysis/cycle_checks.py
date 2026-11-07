"""Checks of the TETDC-1 construction (added 9 Oct 2026).

  python analysis/cycle_checks.py [n_seeds]

1. Recombination bank: number of micro-trips extracted from the 9,000 1 Hz traces that feed both the
   Markov matrix and the micro-trip recombination (same segmentation rule as build_markov.py).
2. Implausible accelerations: micro-trips whose maximum acceleration exceeds 4 m/s^2; whether TETDC-1
   contains any, and how the 99th-percentile acceleration target changes without them.
3. Energy representativeness: energy use of the nominal vehicle (flat ground, each season) for every
   trace of at least 300 s, compared with TETDC-1 (percentile of TETDC-1 among traces, and the
   distance-weighted aggregate of all traces).
4. Seed variability: build_markov.py re-run with seed 42 (published cycle) and other seeds (default 1-10)
   into a scratch folder; composite error, energy use, shape effect and nominal range gap at the central grade.
Output: data/processed/cycle_checks.json
"""
import json, os, sys, subprocess, tempfile
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(BASE, "data", "processed")
sys.path.insert(0, os.path.join(BASE, "drive_cycle")); sys.path.insert(0, os.path.join(BASE, "digital_twin"))
from extract_microtrips import microtrips
from ev_model import EVDigitalTwin
from wltp.cycles import class3
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
tr = pd.read_parquet(os.path.join(P, "microtrips_1hz_sample.parquet"))
twins = {s: EVDigitalTwin(season=s) for s in ("summer", "winter", "shoulder")}
G_MID = 0.0257                     # central mean absolute grade (midpoint of 1.86 % and 3.28 %)
def e100(v, s, g=0.0):
    if g:
        d = np.cumsum(v); g = np.where(d <= d[-1] / 2, g, -g)   # stylised round trip, net elevation 0
    return float(twins[s].simulate(v, g, dt=1.0, soc0=1.0)["kWh_per_100km"])
out = {}
# 1-2 bank and implausible accelerations
mt_amax, mt_acc = [], []
traces = []
for _, g in tr.groupby("trip_id", sort=False):
    v = g["v_ms"].to_numpy()
    if len(v) < 2: continue
    traces.append(v)
    for (_, _, seg) in microtrips(v):
        a = np.gradient(seg, 1.0) if len(seg) > 1 else np.zeros(1)
        mt_amax.append(float(a.max())); mt_acc.append(a)
mt_amax = np.array(mt_amax); bad = mt_amax > 4.0
acc_all = np.concatenate(mt_acc); acc_ok = np.concatenate([a for a, b_ in zip(mt_acc, bad) if not b_])
cyc = pd.read_csv(os.path.join(P, "TETDC_1.csv")).v_ms.to_numpy()
out["bank"] = {"n_traces": len(traces), "n_microtrips": int(len(mt_amax)),
               "n_implausible_amax_gt4": int(bad.sum()), "share_implausible_pct": round(100 * bad.mean(), 2),
               "p99_acc_microtrips_all": round(float(np.percentile(acc_all, 99)), 3),
               "p99_acc_microtrips_without_implausible": round(float(np.percentile(acc_ok, 99)), 3),
               "tetdc_max_acc": round(float(np.gradient(cyc, 1.0).max()), 3)}
# 3 energy representativeness
rows = []
for v in traces:
    if len(v) < 300 or v.sum() < 500: continue
    rows.append({"dist_km": v.sum() / 1000, **{s: e100(v, s) for s in twins}})
df = pd.DataFrame(rows)
cyc_e = {s: e100(cyc, s) for s in twins}
vW = np.array(class3.class_data_b()["cycle"], float) / 3.6
wltc_e = {s: e100(vW, s) for s in twins}
out["energy"] = {"n_traces_ge300s": int(len(df)), "tetdc": {s: round(x, 2) for s, x in cyc_e.items()},
                 "wltc": {s: round(x, 2) for s, x in wltc_e.items()}}
for s in twins:
    w = df.dist_km
    out["energy"][s] = {"trace_median": round(float(df[s].median()), 2),
                        "trace_p10": round(float(df[s].quantile(.1)), 2), "trace_p90": round(float(df[s].quantile(.9)), 2),
                        "distance_weighted_mean": round(float((df[s] * w).sum() / w.sum()), 2),
                        "tetdc_percentile_among_traces": round(float((df[s] < cyc_e[s]).mean() * 100), 1),
                        "tetdc_vs_weighted_mean_pct": round(100 * (cyc_e[s] / ((df[s] * w).sum() / w.sum()) - 1), 1)}
# 4 seed variability
seeds = [42] + [s for s in range(1, N_SEEDS + 1)]    # 42 = the published TETDC-1
sv = []
with tempfile.TemporaryDirectory() as tmp:
    for sd in seeds:
        d = os.path.join(tmp, f"seed{sd}")
        env = dict(os.environ, TETDC_SEED=str(sd), TETDC_OUTDIR=d)
        subprocess.run([sys.executable, "-I", os.path.join(BASE, "drive_cycle", "build_markov.py")], env=env,
                       check=True, stdout=subprocess.DEVNULL)
        st = json.load(open(os.path.join(d, "drive_cycle_stats.json")))
        v = pd.read_csv(os.path.join(d, "TETDC_1.csv")).v_ms.to_numpy()
        sv.append({"seed": sd, "method": st["selected_method"], "composite_err_pct": st["selected"]["composite_err_pct"],
                   "duration_s": int(len(v)), "v_mean_kmh": round(float(v.mean() * 3.6), 2),
                   **{f"kwh_{s}": round(e100(v, s), 2) for s in twins},
                   **{f"shape_{s}": round(e100(v, s) - wltc_e[s], 2) for s in twins},
                   **{f"dR_central_pct_{s}": round(100 * (e100(v, s, G_MID) / wltc_e[s] - 1), 2) for s in twins}})
sdf = pd.DataFrame(sv)
out["seeds"] = {"runs": sv, "summary": {c: {"mean": round(float(sdf[c].mean()), 2), "min": round(float(sdf[c].min()), 2),
                                             "max": round(float(sdf[c].max()), 2), "sd": round(float(sdf[c].std()), 2)}
                                         for c in sdf.columns if c not in ("seed", "method")},
                 "summary_other_seeds": {c: {"mean": round(float(sdf[sdf.seed != 42][c].mean()), 2), "min": round(float(sdf[sdf.seed != 42][c].min()), 2),
                                             "max": round(float(sdf[sdf.seed != 42][c].max()), 2), "sd": round(float(sdf[sdf.seed != 42][c].std()), 2)}
                                         for c in sdf.columns if c not in ("seed", "method")}}
json.dump(out, open(os.path.join(P, "cycle_checks.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "seeds"}, indent=1)); print(json.dumps(out["seeds"]["summary"], indent=1))
