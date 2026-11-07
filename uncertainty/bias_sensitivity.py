"""Sensitivity of the range overestimate to the assumed reconstruction bias (added 9 Oct 2026).

  python uncertainty/bias_sensitivity.py

Re-uses the Monte-Carlo design of uq.py (15 vehicle/terrain inputs, 10,000 paired draws per season,
same seed) and varies only the treatment of the 15 s reconstruction bias b (supplement/bias_factor.py):
  none     : no correction (original analysis)
  half     : correction factor uniform in [0, b]
  fixed    : correction fixed at b
  wide     : correction factor uniform in [0, 2b]  (bias-corrected analysis in the paper)
Output: data/processed/uq_bias_sensitivity.json
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from uq import names_bounds, evaluate, OUTS, BIAS_FILE
B = json.load(open(BIAS_FILE))
res = {"bias_file": os.path.relpath(BIAS_FILE, ROOT)}
for season in ("summer", "winter", "shoulder"):
    names, bounds = names_bounds(season)
    lo = np.array([x[0] for x in bounds]); hi = np.array([x[1] for x in bounds])
    X15 = lo + (hi - lo) * np.random.default_rng(20261003).random((10000, len(bounds)))
    u = np.random.default_rng(20261009).random((10000, 1))
    b = B[season]["b"]
    variants = {"none": None, "half": b * u, "fixed": np.full((10000, 1), b), "wide": 2 * b * u}
    r = {}
    for k, col in variants.items():
        X = X15 if col is None else np.hstack([X15, col])
        Y = np.vstack([evaluate(X[i:i + 2000]) for i in range(0, len(X), 2000)])
        r[k] = {o: dict(mean=round(float(Y[:, j].mean()), 2), p2_5=round(float(np.percentile(Y[:, j], 2.5)), 2),
                        p97_5=round(float(np.percentile(Y[:, j], 97.5)), 2), share_positive=round(float((Y[:, j] > 0).mean()), 3))
                for j, o in enumerate(OUTS) if o in ("shape_effect", "topo_effect", "range_overestimate_pct", "range_overestimate_km")}
    res[season] = {"b": b, "variants": r}
json.dump(res, open(os.path.join(ROOT, "data", "processed", "uq_bias_sensitivity.json"), "w"), indent=1)
for s in ("summer", "winter", "shoulder"):
    for k, v in res[s]["variants"].items():
        m = v["range_overestimate_pct"]; sh = v["shape_effect"]
        print(f"{s:8s} {k:6s} dR {m['mean']:6.2f}% [{m['p2_5']:6.2f}, {m['p97_5']:6.2f}] share>0 {m['share_positive']:.3f} | shape {sh['mean']:5.2f}")
