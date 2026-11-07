"""Does more dynamic driving raise energy use? Test on the 54 open 1 Hz trips.

  python supplement/dynamics_energy.py <out_dir>   (reads <out_dir>/interp_bias_per_trip.csv)

For each season, an ordinary least-squares regression of the energy use computed from the recorded
1 Hz speed (kWh/100 km) on the root-mean-square acceleration (m/s^2) and the mean speed (km/h) of each
trip, so that the acceleration effect is estimated at constant mean speed. Reports the coefficient,
its 95 % interval (5,000 bootstrap resamples of trips, seed 20261107) and the Spearman rank correlation
between RMS acceleration and energy use.
Output: <out_dir>/dynamics_energy.json
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = sys.argv[1]
d = pd.read_csv(os.path.join(OUT, "interp_bias_per_trip.csv")); d = d[d.used == True]
rng = np.random.default_rng(20261107)
def ols(X, y):
    A = np.column_stack([np.ones(len(y)), X]); return np.linalg.lstsq(A, y, rcond=None)[0]
res = {"n_trips": int(len(d))}
X = d[["a_rms_true", "v_mean_kmh_true"]].to_numpy()
for s in ("summer", "winter", "shoulder"):
    y = d[f"kwh100_{s}_true"].to_numpy()
    b = ols(X, y)
    bs = [ols(X[i], y[i])[1] for i in (rng.integers(0, len(y), len(y)) for _ in range(5000))]
    rho, p = spearmanr(d.a_rms_true, y)
    res[s] = {"coef_a_rms_kwh_per_100km_per_ms2": round(float(b[1]), 2),
              "coef_per_0p1_ms2": round(float(b[1]) / 10, 2),
              "coef_boot95": [round(float(x), 2) for x in np.percentile(bs, [2.5, 97.5])],
              "coef_v_mean_kwh_per_100km_per_kmh": round(float(b[2]), 3),
              "spearman_rho_a_rms": round(float(rho), 3), "spearman_p": float(f"{p:.2g}")}
json.dump(res, open(os.path.join(OUT, "dynamics_energy.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
