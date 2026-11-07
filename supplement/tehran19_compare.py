"""Calibrate the metric definitions of Bagheri et al. (2025) on FTP-75 and UDDS, then compute
TETDC-1 with the same definitions and compare with the 19 measured Tehran cycles (Mendeley Data,
doi:10.17632/f7c75syg2f.1, CC BY 4.0). Usage: python tehran19_compare.py <epa_dir> <output_xlsx> <tetdc_csv> <out_dir>"""
import sys, json, itertools, numpy as np, pandas as pd, openpyxl
EPA, XLSX, TETDC, OUT = sys.argv[1:5]
def load(fn):
    d = pd.read_csv(fn, sep="\t", skiprows=2, header=None, names=["t", "mph"])
    return d.mph.values * 1.609344 / 3.6
cyc = {"FTP-75": load(f"{EPA}/ftpcol.txt"), "UDDS": load(f"{EPA}/uddscol.txt")}
ref = {r[0]: r[1:] for r in openpyxl.load_workbook(XLSX, data_only=True)["Characteristic parameters"].iter_rows(min_row=2, values_only=True)}
cols = ["drive","idle","cruise","acc","dec","v_drive","v_trip","v_max","a_pos","a_neg","sd_v","sd_a"]
def metrics(v, v_idle, a_thr, grad):
    a = np.gradient(v) if grad == "central" else np.append(np.diff(v), 0)
    idle = v <= v_idle; mov = ~idle
    acc = mov & (a > a_thr); dec = mov & (a < -a_thr); cru = mov & ~acc & ~dec
    return dict(drive=mov.mean()*100, idle=idle.mean()*100, cruise=cru.mean()*100, acc=acc.mean()*100, dec=dec.mean()*100,
                v_drive=v[mov].mean()*3.6, v_trip=v.mean()*3.6, v_max=v.max()*3.6, a_pos=a[a > a_thr].mean(), a_neg=a[a < -a_thr].mean(),
                sd_v=v.std()*3.6, sd_a=a.std())
best = None
for v_idle, a_thr, grad in itertools.product([0.0, 0.1, 0.28, 0.5, 0.56], [0.0, 0.05, 0.1, 0.15, 0.2], ["central", "forward"]):
    err = 0
    for name, v in cyc.items():
        m = metrics(v, v_idle, a_thr, grad)
        for k in ("idle", "cruise", "acc", "dec", "a_pos", "a_neg", "v_drive", "v_trip"):
            r = ref[name][cols.index(k)]; err += abs(m[k] - r) / abs(r)
    if best is None or err < best[0]: best = (err, v_idle, a_thr, grad)
_, v_idle, a_thr, grad = best
calib = {name: {k: (round(metrics(v, v_idle, a_thr, grad)[k], 2), ref[name][cols.index(k)]) for k in cols} for name, v in cyc.items()}
vt = pd.read_csv(TETDC).v_ms.values
mt = metrics(vt, v_idle, a_thr, grad)
teh = pd.DataFrame([dict(zip(cols, ref[k])) | {"cycle": k} for k in ref if k.startswith("#")])
summary = {"definition": {"idle_speed_ms": v_idle, "accel_threshold_ms2": a_thr, "gradient": grad, "calibration_mean_rel_error": best[0] / 16},
           "calibration_check": calib, "TETDC1": {k: round(float(x), 3) for k, x in mt.items()},
           "tehran19": {k: {"min": float(teh[k].min()), "median": float(teh[k].median()), "max": float(teh[k].max())} for k in cols},
           "TETDC1_percentile_in_tehran19": {k: float((teh[k] < mt[k]).mean()*100) for k in cols}}
json.dump(summary, open(f"{OUT}/tehran19_comparison.json", "w"), indent=1); print(json.dumps(summary, indent=1))
