"""Quantify the bias of reconstructing 1 Hz speed from 15 s GPS positions (method of
drive_cycle/extract_microtrips.py) using open 1 Hz minibus-taxi GPS data (Hull 2024, Mendeley Data V3,
doi:10.17632/xt69cnwh56.3, CC BY 4.0).
Usage: python interp_bias.py <repo_root> <hull_dir> <out_dir>"""
import sys, os, json, importlib.util
import numpy as np, pandas as pd
REPO, HULL, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
spec = importlib.util.spec_from_file_location("em", os.path.join(REPO, "drive_cycle", "extract_microtrips.py"))
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)          # exact pipeline functions
sys.path.insert(0, os.path.join(REPO, "digital_twin")); from ev_model import EVDigitalTwin
split = json.load(open(os.path.join(OUT, "hull_split.json")))      # written by hull_split.py
V_STOP = em.V_STOP

def feats(v):
    a = np.gradient(v, 1.0); pa = a[a > 0.1]
    return dict(duration_s=len(v), distance_km=v.sum()/1000, v_mean_kmh=v.mean()*3.6, v_max_kmh=v.max()*3.6,
                a_max=a.max(), d_max=a.min(), a_rms=np.sqrt((a**2).mean()), a_pos_mean=pa.mean() if len(pa) else 0,
                P_idle=(v < V_STOP).mean()*100, RPA=np.sum(v*np.clip(a,0,None))/max(v.sum(),1e-9))
rows = []
twins = {s: EVDigitalTwin(s) for s in ("summer", "winter", "shoulder")}
for fn in split["raw"]:
    d = pd.read_csv(os.path.join(HULL, fn), encoding="utf-8-sig")
    t = (pd.to_datetime(d["Date"] + " " + d["Time"], format="%m/%d/%Y %H:%M:%S") - pd.Timestamp("1970-01-01")).dt.total_seconds().values
    d = d.assign(t=t).groupby("t", as_index=False).mean(numeric_only=True)        # merge duplicated seconds
    t = d.t.values - d.t.values[0]
    if t[-1] < 300 or np.max(np.diff(t)) > 5: rows.append(dict(file=fn, used=False)); continue
    grid = np.arange(0, int(t[-1]) + 1, 1.0)
    lat, lon = np.interp(grid, t, d.Latitude.values), np.interp(grid, t, d.Longitude.values)
    v_true = em.medfilt3(np.clip(np.interp(grid, t, d.Speed.values) / 3.6, 0, em.V_CAP))      # GPS speed, km/h -> m/s
    v_rec, _ = em.trip_1hz_speed(np.stack([lon[::15], lat[::15]], axis=1))                    # pipeline on 15 s positions
    n = min(len(v_true), len(v_rec)); vt, vr = v_true[:n], v_rec[:n]
    r = dict(file=fn, used=True)
    for k, x in (("true", vt), ("rec", vr)):
        for f, val in feats(x).items(): r[f"{f}_{k}"] = float(val)
        for s, tw in twins.items():
            res = tw.simulate(x, 0.0); r[f"kwh100_{s}_{k}"] = float(res["E_batt_kWh"] / (res["distance_m"]/1000) * 100)
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "interp_bias_per_trip.csv"), index=False)
u = df[df.used == True]
keys = ["distance_km","v_mean_kmh","v_max_kmh","a_max","d_max","a_rms","a_pos_mean","P_idle","RPA","kwh100_summer","kwh100_winter","kwh100_shoulder"]
summ = {"trips_used": int(len(u)), "trips_skipped": int((df.used == False).sum()),
        "total_hours": float(u.duration_s_true.sum()/3600), "total_km_true": float(u.distance_km_true.sum())}
for k in keys:
    rel = (u[f"{k}_rec"] - u[f"{k}_true"]) / u[f"{k}_true"].abs() * 100
    w = u.duration_s_true / u.duration_s_true.sum()
    summ[k] = dict(true_mean=float(np.average(u[f"{k}_true"], weights=w)), rec_mean=float(np.average(u[f"{k}_rec"], weights=w)),
                   rel_bias_pct_median=float(rel.median()), rel_bias_pct_p10=float(rel.quantile(.1)), rel_bias_pct_p90=float(rel.quantile(.9)))
json.dump(summ, open(os.path.join(OUT, "interp_bias_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
