"""Control: same PCHIP pipeline on 1 s positions (isolates the effect of 15 s subsampling),
plus subsampling at 5 s and 10 s, and stratification by driving dynamics.
Usage: python interp_control.py <repo_root> <hull_dir> <out_dir>"""
import sys, os, json, importlib.util
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
REPO, HULL, OUT = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("em", os.path.join(REPO, "drive_cycle", "extract_microtrips.py"))
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)
sys.path.insert(0, os.path.join(REPO, "digital_twin")); from ev_model import EVDigitalTwin
split = json.load(open(os.path.join(OUT, "hull_split.json")))      # written by hull_split.py
tw = EVDigitalTwin("summer")
def rec(lon, lat, step):                      # identical to em.trip_1hz_speed but with a variable step
    steps = em.haversine_step(lon[:-1], lat[:-1], lon[1:], lat[1:])
    s = np.concatenate([[0.0], np.cumsum(steps)]); tt = step * np.arange(len(lon))
    pc = PchipInterpolator(tt, s); t1 = np.arange(int(tt[-1]) + 1, dtype=float)
    return em.medfilt3(np.clip(pc(t1 + 0.5) - pc(t1 - 0.5), 0.0, em.V_CAP))
def f(v):
    a = np.gradient(v); r = tw.simulate(v, 0.0)
    return dict(a_rms=np.sqrt((a**2).mean()), P_idle=(v < em.V_STOP).mean()*100, kwh=r["E_batt_kWh"]/(r["distance_m"]/1000)*100,
                v_mean=v.mean()*3.6)
rows = []
for fn in split["raw"]:
    d = pd.read_csv(os.path.join(HULL, fn), encoding="utf-8-sig")
    t = (pd.to_datetime(d["Date"] + " " + d["Time"], format="%m/%d/%Y %H:%M:%S") - pd.Timestamp("1970-01-01")).dt.total_seconds().values
    d = d.assign(t=t).groupby("t", as_index=False).mean(numeric_only=True); t = d.t.values - d.t.values[0]
    if t[-1] < 300 or np.max(np.diff(t)) > 5: continue
    g = np.arange(0, int(t[-1]) + 1, 1.0); lat, lon = np.interp(g, t, d.Latitude.values), np.interp(g, t, d.Longitude.values)
    v_dop = em.medfilt3(np.clip(np.interp(g, t, d.Speed.values) / 3.6, 0, em.V_CAP))
    out = {"file": fn, **{f"{k}_doppler": val for k, val in f(v_dop).items()}}
    for step in (1, 5, 10, 15):
        v = rec(lon[::step], lat[::step], step); n = min(len(v), len(v_dop))
        out.update({f"{k}_pos{step}": val for k, val in f(v[:n]).items()})
    rows.append(out)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "interp_control_per_trip.csv"), index=False)
res = {"trips": len(df)}
for k in ("a_rms", "P_idle", "kwh", "v_mean"):
    res[k] = {"doppler": float(df[f"{k}_doppler"].mean())}
    for step in (1, 5, 10, 15):
        res[k][f"pos{step}_mean"] = float(df[f"{k}_pos{step}"].mean())
        res[k][f"pos{step}_vs_pos1_median_pct"] = float(((df[f"{k}_pos{step}"] - df[f"{k}_pos1"]) / df[f"{k}_pos1"].abs() * 100).median())
# energy bias of 15 s vs 1 s positions, stratified by true dynamics (a_rms of 1 s positions)
df["bias15"] = (df.kwh_pos15 - df.kwh_pos1) / df.kwh_pos1 * 100
df["tercile"] = pd.qcut(df.a_rms_pos1, 3, labels=["low", "mid", "high"])
res["energy_bias_15s_by_dynamics"] = {str(k): {"a_rms_range": [float(g.a_rms_pos1.min()), float(g.a_rms_pos1.max())],
                                       "median_bias_pct": float(g.bias15.median()), "n": int(len(g))} for k, g in df.groupby("tercile", observed=True)}
res["corr_bias15_vs_arms"] = float(np.corrcoef(df.a_rms_pos1, df.bias15)[0, 1])
json.dump(res, open(os.path.join(OUT, "interp_control_summary.json"), "w"), indent=1); print(json.dumps(res, indent=1))
