"""Ensemble Monte-Carlo analysis over construction runs of the cycle (added November 2026).

  python uncertainty/uq_ensemble.py

The cycle construction (drive_cycle/build_markov.py) is repeated with seed 42 (the published TETDC-1)
and seeds 1-10. Each of the 11 cycles is saved in data/processed/ensemble/seed<k>/TETDC_1.csv.
Every cycle is then evaluated on the same 10,000 paired Monte-Carlo draws as uq.py (modes mc and mc_bc,
identical input matrices), so the pooled sample of 110,000 draws per season combines parameter
uncertainty, reconstruction-bias uncertainty (mc_bc) and construction uncertainty with equal weight per
cycle. A second pool uses only the cycles whose composite error is below 10 %.
For each pool, the variance of every output is split by the law of total variance into a between-cycle
part (variance of the cycle means) and a within-cycle part (mean of the variances within each cycle);
"share_between" is the fraction of the pooled variance that is due to the construction of the cycle.
Output: data/processed/uq_ensemble.json
"""
import json, os, subprocess, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import uq
P = os.path.join(ROOT, "data", "processed"); ED = os.path.join(P, "ensemble")
SEEDS = [42] + list(range(1, 11))
ERR_MAX = 10.0
OUT_IDX = {o: j for j, o in enumerate(uq.OUTS)}
KEEP = ["shape_effect", "topo_effect", "range_overestimate_pct", "range_overestimate_km"]

cycles = {}
for sd in SEEDS:
    d = os.path.join(ED, f"seed{sd}")
    if not os.path.exists(os.path.join(d, "TETDC_1.csv")):
        env = dict(os.environ, TETDC_SEED=str(sd), TETDC_OUTDIR=d)
        subprocess.run([sys.executable, "-I", os.path.join(ROOT, "drive_cycle", "build_markov.py")], env=env,
                       check=True, stdout=subprocess.DEVNULL)
    st = json.load(open(os.path.join(d, "drive_cycle_stats.json")))
    cycles[sd] = {"v": pd.read_csv(os.path.join(d, "TETDC_1.csv")).v_ms.to_numpy(),
                  "err": st["selected"]["composite_err_pct"]}
ref = pd.read_csv(os.path.join(P, "TETDC_1.csv")).v_ms.to_numpy()
assert np.array_equal(ref, cycles[42]["v"]), "seed-42 rebuild differs from the published TETDC-1"

def stats(y):
    return dict(mean=round(float(y.mean()), 2), p2_5=round(float(np.percentile(y, 2.5)), 2),
                p97_5=round(float(np.percentile(y, 97.5)), 2), share_positive=round(float((y > 0).mean()), 3))

res = {"seeds": SEEDS, "composite_err_pct": {str(k): c["err"] for k, c in cycles.items()},
       "subset_err_lt_10": [k for k, c in cycles.items() if c["err"] < ERR_MAX], "seasons": {}}
for season in ("summer", "winter", "shoulder"):
    res["seasons"][season] = {}
    for mode in ("mc", "mc_bc"):
        uq.run(mode, season)                                   # makes sure the shared input matrix exists
        X = np.load(os.path.join(uq.OUTDIR, f"{mode}_{season}_X.npy"))
        per = {}
        for sd, c in cycles.items():
            f = os.path.join(ED, f"seed{sd}", f"{mode}_{season}_Y.npy")
            if not os.path.exists(f):
                uq.vT = c["v"]; uq.dT = np.cumsum(c["v"])
                np.save(f, np.vstack([uq.evaluate(X[i:i + 2000]) for i in range(0, len(X), 2000)]))
            per[sd] = np.load(f)
        r = {"per_cycle": {str(sd): {o: stats(Y[:, OUT_IDX[o]]) for o in KEEP} for sd, Y in per.items()}}
        for tag, ks in (("pooled_all", SEEDS), ("pooled_err_lt_10", res["subset_err_lt_10"])):
            Y = np.vstack([per[k] for k in ks])
            r[tag] = {o: stats(Y[:, OUT_IDX[o]]) for o in KEEP}; r[tag]["n_draws"] = int(len(Y))
            r[tag]["variance_split"] = {}
            for o in ("shape_effect", "range_overestimate_pct"):
                cols = [per[k][:, OUT_IDX[o]] for k in ks]
                between = float(np.var([c.mean() for c in cols])); within = float(np.mean([c.var() for c in cols]))
                r[tag]["variance_split"][o] = dict(total=round(between + within, 3), between=round(between, 3),
                                                   within=round(within, 3), share_between=round(between / (between + within), 4))
        res["seasons"][season][mode] = r
        print(season, mode, "pooled dR", r["pooled_all"]["range_overestimate_pct"], flush=True)
uq.vT = ref; uq.dT = np.cumsum(ref)
json.dump(res, open(os.path.join(P, "uq_ensemble.json"), "w"), indent=1)
