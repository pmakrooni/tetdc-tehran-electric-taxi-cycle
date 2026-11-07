"""Aggregate energy bias of the 15 s -> 1 Hz reconstruction, per season, from interp_bias_per_trip.csv.
b = (E_true / E_rec) - 1, where E is total energy over total distance of all usable trips
(true = 1 Hz GPS Doppler speed, rec = pipeline reconstruction from 15 s positions).
95 % interval from 5000 bootstrap resamples of trips (seed 20261008).
uncertainty/uq.py samples the correction factor from U(0, 2b): centred on the measured value,
allowing for no bias (smoother Porto driving) up to twice the measured bias.
Usage: python bias_factor.py <out_dir>   (reads <out_dir>/interp_bias_per_trip.csv)"""
import sys, os, json
import numpy as np, pandas as pd
OUT = sys.argv[1]
d = pd.read_csv(os.path.join(OUT, "interp_bias_per_trip.csv")); d = d[d.used == True]
rng = np.random.default_rng(20261008); res = {}
for s in ("summer", "winter", "shoulder"):
    Et = (d[f"kwh100_{s}_true"] * d.distance_km_true).values; Er = (d[f"kwh100_{s}_rec"] * d.distance_km_rec).values
    Dt, Dr = d.distance_km_true.values, d.distance_km_rec.values
    agg = lambda i: (Et[i].sum() / Dt[i].sum()) / (Er[i].sum() / Dr[i].sum()) - 1
    allidx = np.arange(len(d)); b = float(agg(allidx))
    bs = [agg(rng.integers(0, len(d), len(d))) for _ in range(5000)]
    res[s] = {"b": round(b, 4), "boot95": [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])],
              "uq_bounds": [0.0, round(2 * b, 4)], "n_trips": int(len(d))}
json.dump(res, open(os.path.join(OUT, "bias_factor.json"), "w"), indent=1); print(json.dumps(res, indent=1))
