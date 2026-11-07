"""TETDC — Deep drive-cycle validation (master-prompt sections 7, 8 & 13).

Section 7 — validate TETDC-1 against real-data targets on 15 kinematic
features (prompt requires 12+), plus distribution-level checks:
  * RMSE between normalized speed histograms (cycle vs fleet 1 Hz),
    reported in two frames: raw pooled traces AND the filtered micro-trip
    concatenation (the construction frame)
  * RMSE of the joint velocity-acceleration PDF
  * two-sample Kolmogorov-Smirnov tests (speed, acceleration)

TARGET DEFINITIONS (consistent with build_markov.py / extract_microtrips.py):
  * duration-weighted feature means over the micro-trip bank:
    v_mean, P_idle, a_rms, v_std, a_pos_mean, a_neg_mean
  * extremes as 1 Hz percentiles of the pooled trace sample:
    v_max = p99(v), a_max = p99(a), d_max = p01(a)
  * per-micro-trip extras from the SAME filtered segmentation
    (>=10 s, >=25 m) as the construction bank:
    PKE (distance-weighted), stops_per_km = n_microtrips / distance,
    P_accel / P_decel / P_cruise / v_mean_mov (duration-weighted)

Section 8 — representativeness cross-validation on UNSEEN segments:
  * route-level split: trips split 70/30 by deterministic md5(trip_id).
    (i)   TETDC-1 (built on all data) scored against TEST-only targets;
    (ii)  a cycle rebuilt from TRAIN-only traces (Method B recombination +
          hill-climbing, reduced budget) scored against the same targets;
    (iii) bootstrap (200 resamples) distribution of TEST-target errors.
  * time-of-day split: peak hours (7-9, 16-19 UTC) vs off-peak.
    TETDC-1 scored against each regime; an off-peak-only rebuild is scored
    against peak targets (hard transfer test).

Section 13 (baseline 3) — documented characteristics of existing Tehran
drive cycles from the literature (abstract values only; full traces are
not publicly downloadable, so energy simulation for them is impossible
without fabrication, which section 32 prohibits).

Output: data/processed/validation_results.json + TETDC_CV_route_rebuild.csv
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

SEED = 42
rng = np.random.default_rng(SEED)


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.path.join(BASE, "data", "processed")
FIGTRACES = os.path.join(OUTD, "microtrips_1hz_sample.parquet")
FEATS = os.path.join(OUTD, "microtrips_features.parquet")
CYCLE = os.path.join(OUTD, "TETDC_1.csv")

V_STOP = 0.5          # m/s stop threshold (extract_microtrips convention)
A_EPS = 0.1           # m/s^2 accel/decel regime threshold
PEAK_HOURS = {7, 8, 9, 16, 17, 18, 19}   # UTC (Porto timestamps)


# --------------------------------------------------------------- helpers
def microtrips(v, min_s=10, min_m=25):
    """Same segmentation rule as extract_microtrips (import-free copy)."""
    n = len(v)
    stopmask = v < V_STOP
    ev, i = [], 0
    while i < n:
        if stopmask[i]:
            j = i
            while j < n and stopmask[j]:
                j += 1
            if j - i >= 3:
                ev.append((i, j))
            i = j
        else:
            i += 1
    bounds = [0] + [e[0] for e in ev] + [n]
    out = []
    for b in range(len(bounds) - 1):
        a0, a1 = bounds[b], bounds[b + 1]
        seg = v[a0:a1]
        if len(seg) < min_s or float(seg.sum()) < min_m:
            continue
        out.append((a0, a1, seg))
    return out


def trace_features(v):
    """15 kinematic features of a 1 Hz speed trace (cycle or micro-trip)."""
    acc = np.gradient(v, 1.0)
    pos_a, neg_a = acc[acc > A_EPS], acc[acc < -A_EPS]
    mov = v >= V_STOP
    dv2 = np.diff(v ** 2)
    pke = float(np.sum(dv2[dv2 > 0]) / (2.0 * max(v.sum(), 1.0)))
    stopmask = v < V_STOP
    stops, i, n = 0, 0, len(v)
    while i < n:
        if stopmask[i]:
            j = i
            while j < n and stopmask[j]:
                j += 1
            if j - i >= 3:
                stops += 1
            i = j
        else:
            i += 1
    return {
        "duration_s": float(len(v)),
        "distance_m": float(v.sum()),
        "v_mean": float(v.mean()),
        "v_max": float(v.max()),
        "v_std": float(v.std()),
        "a_max": float(acc.max()),
        "d_max": float(acc.min()),
        "a_pos_mean": float(pos_a.mean()) if len(pos_a) else 0.0,
        "a_neg_mean": float(neg_a.mean()) if len(neg_a) else 0.0,
        "P_idle": float((~mov).mean() * 100),
        "a_rms": float(np.sqrt((acc ** 2).mean())),
        "PKE": pke,
        "stops_per_km": stops / (v.sum() / 1000.0),
        "P_accel": float((acc > A_EPS).mean() * 100),
        "P_decel": float((acc < -A_EPS).mean() * 100),
        "P_cruise": float(((np.abs(acc) <= A_EPS) & mov).mean() * 100),
        "v_mean_mov": float(v[mov].mean()) if mov.any() else 0.0,
    }


FEAT_KEYS = ["v_mean", "v_max", "v_std", "a_max", "d_max", "a_pos_mean",
             "a_neg_mean", "P_idle", "a_rms", "PKE", "stops_per_km",
             "P_accel", "P_decel", "P_cruise", "v_mean_mov"]

EXTRA_KEYS = ["PKE", "stops_per_km", "P_accel", "P_decel", "P_cruise",
              "v_mean_mov"]


def microtrip_bank_extras(bank):
    """Per-micro-trip extra features from a 1 Hz trace bank using the SAME
    filtered segmentation as the construction bank. One micro-trip = one
    stop-to-stop cycle, hence exactly one stop event per micro-trip."""
    rows = []
    for tid, g in bank.groupby("trip_id", sort=False):
        v = g["v_ms"].to_numpy(float)
        for (_, _, seg) in microtrips(v):
            acc = np.gradient(seg, 1.0)
            dv2 = np.diff(seg ** 2)
            pke = float(np.sum(dv2[dv2 > 0]) / (2.0 * max(seg.sum(), 1.0)))
            mov = seg >= V_STOP
            rows.append({
                "trip_id": tid, "duration_s": len(seg),
                "distance_m": float(seg.sum()), "PKE": pke, "stops": 1.0,
                "P_accel": float((acc > A_EPS).mean() * 100),
                "P_decel": float((acc < -A_EPS).mean() * 100),
                "P_cruise": float(((np.abs(acc) <= A_EPS) & mov).mean() * 100),
                "v_mean_mov": float(seg[mov].mean()) if mov.any() else 0.0,
            })
    return pd.DataFrame(rows)


def extras_targets(extras):
    """PKE distance-weighted; stops_per_km = n / total distance; regime
    shares duration-weighted (construction frame). v_mean_mov is a pooled
    trace statistic and is handled in fleet_targets."""
    d = extras["distance_m"].to_numpy(float)
    t = extras["duration_s"].to_numpy(float)
    return {
        "PKE": float(np.average(extras["PKE"], weights=d)),
        "stops_per_km": float(len(extras) / (d.sum() / 1000.0)),
        "P_accel": float(np.average(extras["P_accel"], weights=t)),
        "P_decel": float(np.average(extras["P_decel"], weights=t)),
        "P_cruise": float(np.average(extras["P_cruise"], weights=t)),
    }


def pooled_va(bank):
    v = bank["v_ms"].to_numpy(float)
    a = np.concatenate([np.gradient(g["v_ms"].to_numpy(float), 1.0)
                        for _, g in bank.groupby("trip_id", sort=False)])
    return v, a


def fleet_targets(feats_df, pool_v, pool_a, extras):
    """All 15 targets, each on its consistent definition frame.

    Duration-weighted per-micro-trip means (mathematically equal to the
    pooled share over the bank): v_mean, P_idle, a_rms, a_pos_mean,
    a_neg_mean. Whole-trace statistics pooled over the 1 Hz sample:
    v_std, v_mean_mov. Percentile extremes: v_max/a_max/d_max. Filtered
    micro-trip-bank extras: PKE, stops_per_km, regime shares.
    """
    w = feats_df["duration_s"].to_numpy(float)
    out = {}
    for k in ["v_mean", "P_idle", "a_rms", "a_pos_mean", "a_neg_mean"]:
        out[k] = float(np.average(feats_df[k].to_numpy(float), weights=w))
    out["v_max"] = float(np.percentile(pool_v, 99))
    out["a_max"] = float(np.percentile(pool_a, 99))
    out["d_max"] = float(np.percentile(pool_a, 1))
    mov = pool_v >= V_STOP
    out["v_std"] = float(np.std(pool_v))
    out["v_mean_mov"] = float(pool_v[mov].mean())
    out.update(extras_targets(extras))
    return out


def rel_errors(cycle_f, targets):
    rows = []
    for k in FEAT_KEYS:
        t, c = targets[k], cycle_f[k]
        ta = abs(t) if k == "d_max" else t
        ca = abs(c) if k == "d_max" else c
        rows.append({"feature": k, "target": round(ta, 4),
                     "cycle": round(ca, 4),
                     "rel_err_pct": round(100 * abs(ca - ta) / max(abs(ta), 1e-9), 2)})
    return rows


def composite(rows):
    return float(np.mean([r["rel_err_pct"] for r in rows]))


# --------------------------------------------------------------- load data
print("loading ...", flush=True)
feats = pd.read_parquet(FEATS)
tr = pd.read_parquet(FIGTRACES)
cyc = pd.read_csv(CYCLE)
v_cycle = cyc["v_ms"].to_numpy(float)
acc_cycle = np.gradient(v_cycle, 1.0)

v_pool, acc_pool = pooled_va(tr)

# per-micro-trip extras for the whole trace bank (one segmentation pass)
print("segmenting trace bank for extra-feature targets ...", flush=True)
extras_full = microtrip_bank_extras(tr)
# micro-trip concatenation = the construction frame signal
mt_cat = np.concatenate([
    seg for _, g in tr.groupby("trip_id", sort=False)
    for (_, _, seg) in microtrips(g["v_ms"].to_numpy(float))
])
acc_mt_cat = np.gradient(mt_cat, 1.0)

# --------------------------------------------------------------- § 7
print("== section 7: 15-feature validation ==", flush=True)
targets_full = fleet_targets(feats, v_pool, acc_pool, extras_full)
cycle_f = trace_features(v_cycle)
rows7 = rel_errors(cycle_f, targets_full)
comp7 = composite(rows7)

# distribution-level metrics — dual frame
bins_v = np.arange(0, 34.0, 1.0)
ab = np.arange(-3.0, 3.25, 0.25)


def pdf_rmse(x_c, x_f, bins):
    hc = np.histogram(x_c, bins=bins)[0].astype(float)
    hf = np.histogram(x_f, bins=bins)[0].astype(float)
    pc, pf = hc / hc.sum(), hf / hf.sum()
    return float(np.sqrt(np.mean((pc - pf) ** 2)))


def va_rmse(v_c, a_c, v_f, a_f):
    ac_c = np.clip(a_c, -3, 3)
    ac_f = np.clip(a_f, -3, 3)
    hca = np.histogram2d(v_c, ac_c, bins=[bins_v, ab])[0]
    hfa = np.histogram2d(v_f, ac_f, bins=[bins_v, ab])[0]
    pca_, pfa_ = hca / hca.sum(), hfa / hfa.sum()
    return float(np.sqrt(np.mean((pca_ - pfa_) ** 2)))


speed_rmse_raw = pdf_rmse(v_cycle, v_pool, bins_v)
speed_rmse_mt = pdf_rmse(v_cycle, mt_cat, bins_v)
va_rmse_raw = va_rmse(v_cycle, acc_cycle, v_pool, acc_pool)
va_rmse_mt = va_rmse(v_cycle, acc_cycle, mt_cat, acc_mt_cat)
ks_v_raw = sps.ks_2samp(v_cycle, v_pool)
ks_v_mt = sps.ks_2samp(v_cycle, mt_cat)
ks_a_mt = sps.ks_2samp(np.clip(acc_cycle, -3, 3),
                       np.clip(acc_mt_cat, -3, 3))
print(f"  composite |err| = {comp7:.2f}% | speedPDF RMSE raw/mt = "
      f"{speed_rmse_raw:.5f}/{speed_rmse_mt:.5f} | KS D(v) raw/mt = "
      f"{ks_v_raw.statistic:.4f}/{ks_v_mt.statistic:.4f}", flush=True)

# --------------------------------------------------------------- § 8 route
print("== section 8: route-level cross-validation ==", flush=True)


def split_ids(series):
    train, test = [], []
    for tid in series.unique():
        h = int(hashlib.md5(str(tid).encode()).hexdigest(), 16) % 10
        (train if h < 7 else test).append(tid)
    return set(train), set(test)


train_ids, test_ids = split_ids(feats["trip_id"])
f_tr = feats[feats["trip_id"].isin(train_ids)]
f_te = feats[feats["trip_id"].isin(test_ids)]
g_tr = tr[tr["trip_id"].isin(train_ids)]
g_te = tr[tr["trip_id"].isin(test_ids)]
extras_te = extras_full[extras_full["trip_id"].isin(test_ids)]
v_te_pool, acc_te_pool = pooled_va(g_te)
targets_te = fleet_targets(f_te, v_te_pool, acc_te_pool, extras_te)

rows8_route = rel_errors(cycle_f, targets_te)
comp8_route = composite(rows8_route)

# rebuild a cycle from TRAIN-only traces (Method B, reduced budget)
train_trace_bank = []
for _, g in g_tr.groupby("trip_id", sort=False):
    v = g["v_ms"].to_numpy(float)
    for (_, _, seg) in microtrips(v):
        train_trace_bank.append(seg)
bank_lens = np.array([len(s) for s in train_trace_bank])
print(f"  train trace bank: {len(train_trace_bank):,} micro-trips", flush=True)

w_tr = f_tr["duration_s"].to_numpy(float)
v_tr_pool, acc_tr_pool = pooled_va(g_tr)
opt_keys = ["v_mean", "v_max", "a_max", "d_max", "P_idle", "a_rms"]
tgt_opt = {
    "v_mean": float(np.average(f_tr["v_mean"], weights=w_tr)),
    "P_idle": float(np.average(f_tr["P_idle"], weights=w_tr)),
    "a_rms": float(np.average(f_tr["a_rms"], weights=w_tr)),
    "v_max": float(np.percentile(v_tr_pool, 99)),
    "a_max": float(np.percentile(acc_tr_pool, 99)),
    "d_max": float(np.percentile(acc_tr_pool, 1)),
}


def opt_err(f):
    e = []
    for k in opt_keys:
        t = abs(tgt_opt[k]) if k == "d_max" else tgt_opt[k]
        c = abs(f[k]) if k == "d_max" else f[k]
        e.append(abs(c - t) / max(abs(t), 1e-6))
    return float(np.mean(e))


T_LO, T_HI = 1700, 1900
r2 = np.random.default_rng(SEED + 11)
best_parts, best_err = None, 1e9
for _ in range(700):
    parts, tot = [], 0
    while tot < T_LO:
        k = int(r2.integers(len(train_trace_bank)))
        parts.append(k)
        tot += bank_lens[k]
        if tot > 1.15 * T_HI:
            break
    if not (T_LO <= tot <= T_HI + 60):
        continue
    v = np.concatenate([train_trace_bank[k] for k in parts])
    e = opt_err(trace_features(v))
    if e < best_err:
        best_parts, best_err = parts, e
tot = int(sum(bank_lens[k] for k in best_parts))
improved = 0
for it in range(3500):
    if not best_parts:
        break
    pos = int(r2.integers(len(best_parts)))
    k = int(r2.integers(len(train_trace_bank)))
    nt = tot - bank_lens[best_parts[pos]] + bank_lens[k]
    if not (T_LO <= nt <= T_HI + 60):
        continue
    trial = best_parts.copy()
    trial[pos] = k
    v = np.concatenate([train_trace_bank[j] for j in trial])
    e = opt_err(trace_features(v))
    if e < best_err:
        best_parts, tot, best_err, improved = trial, nt, e, improved + 1
v_cv = np.concatenate([train_trace_bank[j] for j in best_parts])
cv_f = trace_features(v_cv)
rows8_rebuild = rel_errors(cv_f, targets_te)
comp8_rebuild = composite(rows8_rebuild)
print(f"  train-rebuild opt_err {best_err*100:.2f}% -> TEST composite "
      f"{comp8_rebuild:.2f}% ({improved} swaps)", flush=True)

# bootstrap distribution of TETDC-1 errors on TEST targets
print("  bootstrap ...", flush=True)
n_boot = 200
te_w = f_te["duration_s"].to_numpy(float)
te_mat = f_te[["v_mean", "v_std", "a_pos_mean", "a_neg_mean", "P_idle",
               "a_rms"]].to_numpy(float)
ex_te = extras_te[["PKE", "P_accel", "P_decel", "P_cruise", "v_mean_mov",
                   "distance_m", "duration_s"]].to_numpy(float)
boot_rows = {k: [] for k in FEAT_KEYS}
boot_comp = []
for b in range(n_boot):
    idx = rng.integers(0, len(f_te), size=len(f_te))
    w = te_w[idx]
    tgt_b = dict(zip(["v_mean", "a_pos_mean", "a_neg_mean",
                      "P_idle", "a_rms"],
                     np.average(te_mat[idx][:, [0, 2, 3, 4, 5]], axis=0,
                                weights=w)))
    for k in ["v_max", "a_max", "d_max", "v_std", "v_mean_mov"]:
        tgt_b[k] = targets_te[k]   # pooled / percentile targets: fixed
    ie = rng.integers(0, len(extras_te), size=len(extras_te))
    es = ex_te[ie]
    d_w = es[:, 5]
    t_w = es[:, 6]
    tgt_b["PKE"] = float(np.average(es[:, 0], weights=d_w))
    tgt_b["stops_per_km"] = float(len(ie) / (d_w.sum() / 1000.0))
    for j, k in enumerate(["P_accel", "P_decel", "P_cruise"]):
        tgt_b[k] = float(np.average(es[:, j + 1], weights=t_w))
    rows = rel_errors(cycle_f, tgt_b)
    for r in rows:
        boot_rows[r["feature"]].append(r["rel_err_pct"])
    boot_comp.append(composite(rows))

boot_summary = {
    k: {"p5": round(float(np.percentile(x, 5)), 2),
        "p50": round(float(np.percentile(x, 50)), 2),
        "p95": round(float(np.percentile(x, 95)), 2)}
    for k, x in boot_rows.items()
}
boot_comp_s = {p: round(float(np.percentile(boot_comp, q)), 2)
               for p, q in [("p5", 5), ("p50", 50), ("p95", 95)]}

# --------------------------------------------------------------- § 8 time
print("== section 8: time-of-day cross-validation ==", flush=True)
peak_trip_ids = set(feats[feats["hour"].isin(PEAK_HOURS)]["trip_id"])
f_pk = feats[feats["hour"].isin(PEAK_HOURS)]
f_op = feats[~feats["hour"].isin(PEAK_HOURS)]
tr_pk = tr[tr["trip_id"].isin(peak_trip_ids)]
tr_op = tr[~tr["trip_id"].isin(peak_trip_ids)]
extras_pk = extras_full[extras_full["trip_id"].isin(peak_trip_ids)]
extras_op = extras_full[~extras_full["trip_id"].isin(peak_trip_ids)]
v_pk, a_pk = pooled_va(tr_pk)
v_op, a_op = pooled_va(tr_op)
targets_pk = fleet_targets(f_pk, v_pk, a_pk, extras_pk)
targets_op = fleet_targets(f_op, v_op, a_op, extras_op)

rows8_peak = rel_errors(cycle_f, targets_pk)
rows8_offpeak = rel_errors(cycle_f, targets_op)
comp8_peak, comp8_offpeak = composite(rows8_peak), composite(rows8_offpeak)

# off-peak-only rebuild -> transfer test against PEAK targets
op_bank = []
for _, g in tr_op.groupby("trip_id", sort=False):
    v = g["v_ms"].to_numpy(float)
    for (_, _, seg) in microtrips(v):
        op_bank.append(seg)
op_lens = np.array([len(s) for s in op_bank])
w_op = f_op["duration_s"].to_numpy(float)
tgt_op6 = {
    "v_mean": float(np.average(f_op["v_mean"], weights=w_op)),
    "P_idle": float(np.average(f_op["P_idle"], weights=w_op)),
    "a_rms": float(np.average(f_op["a_rms"], weights=w_op)),
    "v_max": float(np.percentile(v_op, 99)),
    "a_max": float(np.percentile(a_op, 99)),
    "d_max": float(np.percentile(a_op, 1)),
}


def opt_err_op(f):
    e = []
    for k in opt_keys:
        t = abs(tgt_op6[k]) if k == "d_max" else tgt_op6[k]
        c = abs(f[k]) if k == "d_max" else f[k]
        e.append(abs(c - t) / max(abs(t), 1e-6))
    return float(np.mean(e))


r3 = np.random.default_rng(SEED + 23)
parts2, err2 = None, 1e9
for _ in range(700):
    parts, tot = [], 0
    while tot < T_LO:
        k = int(r3.integers(len(op_bank)))
        parts.append(k)
        tot += op_lens[k]
        if tot > 1.15 * T_HI:
            break
    if not (T_LO <= tot <= T_HI + 60):
        continue
    v = np.concatenate([op_bank[k] for k in parts])
    e = opt_err_op(trace_features(v))
    if e < err2:
        parts2, err2 = parts, e
tot = int(sum(op_lens[k] for k in parts2))
for it in range(2500):
    if not parts2:
        break
    pos = int(r3.integers(len(parts2)))
    k = int(r3.integers(len(op_bank)))
    nt = tot - op_lens[parts2[pos]] + op_lens[k]
    if not (T_LO <= nt <= T_HI + 60):
        continue
    trial = parts2.copy()
    trial[pos] = k
    v = np.concatenate([op_bank[j] for j in trial])
    e = opt_err_op(trace_features(v))
    if e < err2:
        parts2, tot, err2 = trial, nt, e
v_op_cv = np.concatenate([op_bank[j] for j in parts2])
opcv_f = trace_features(v_op_cv)
rows8_transfer = rel_errors(opcv_f, targets_pk)
comp8_transfer = composite(rows8_transfer)
print(f"  off-peak rebuild -> peak transfer composite {comp8_transfer:.2f}%",
      flush=True)

# --------------------------------------------------------------- § 13 data
tehran_cycles = {
    "note": ("Values quoted from the published abstracts; the full 1 Hz "
             "speed traces are not publicly downloadable, so an energy "
             "simulation for these cycles cannot be run without fabricating "
             "the traces (prohibited by section 32)."),
    "fotouhi_2013_tcdc": {
        "ref": ("Fotouhi, A., Montazeri-Gh, M. (2013). Tehran driving cycle "
                "development using the k-means clustering method. Scientia "
                "Iranica 20(2), 286-293. DOI 10.1016/j.scient.2013.04.001"),
        "vehicle_class": "private passenger car (city + suburbs)",
        "duration_s": 1533, "v_mean_kmh": 33.83, "distance_km": 14.41,
        "P_idle_pct": 15.26,
    },
    "montazeri_2007": {
        "ref": ("Montazeri-Gh, M., Naghizadeh, M. (2007). Development of the "
                "Tehran car driving cycle. Int. J. Environment and Pollution "
                "30(1). DOI 10.1504/IJEP.2007.014506"),
        "vehicle_class": "private passenger car",
        "duration_s": None, "v_mean_kmh": None, "distance_km": None,
        "P_idle_pct": None,
    },
    "mafi_2022": {
        "ref": ("Mafi, S., Kakaee, A.H., Mashadi, B., Moosavian, A., "
                "Abdolmajidi, E. (2022). Developing local driving cycle for "
                "accurate vehicular CO2 monitoring: A case study of Tehran. "
                "Journal of Cleaner Production 331, 130176. DOI "
                "10.1016/j.jclepro.2021.130176"),
        "vehicle_class": "passenger car (CO2 monitoring focus)",
        "duration_s": None, "v_mean_kmh": None, "distance_km": None,
        "P_idle_pct": None,
    },
    "tetdc_1_this_study": {
        "vehicle_class": "electric taxi (Aion V Plus, transferred pattern)",
        "duration_s": int(cycle_f["duration_s"]),
        "v_mean_kmh": round(cycle_f["v_mean"] * 3.6, 2),
        "distance_km": round(cycle_f["distance_m"] / 1000.0, 2),
        "P_idle_pct": round(cycle_f["P_idle"], 2),
    },
}

# --------------------------------------------------------------- save
result = {
    "seed": SEED,
    "target_definitions": {
        "duration_weighted_features": ["v_mean", "P_idle", "a_rms",
                                        "a_pos_mean", "a_neg_mean"],
        "pooled_trace_statistics": "v_std and v_mean_mov over the pooled 1 Hz trace sample",
        "percentile_extremes": "v_max = p99(v), a_max = p99(a), d_max = p01(a) of pooled 1 Hz traces",
        "microtrip_extras": ("PKE distance-weighted; stops_per_km = n_microtrips / distance; "
                             "P_accel/P_decel/P_cruise duration-weighted; "
                             "same >=10 s / >=25 m filtered segmentation as construction"),
    },
    "sec7_feature_validation": {
        "n_features": len(FEAT_KEYS),
        "targets": {k: round(v, 4) for k, v in targets_full.items()},
        "cycle_features": {k: round(cycle_f[k], 4) for k in FEAT_KEYS},
        "feature_rows": rows7,
        "composite_err_pct": round(comp7, 2),
        "max_err_pct": round(max(r["rel_err_pct"] for r in rows7), 2),
        "n_within_10pct": sum(r["rel_err_pct"] <= 10 for r in rows7),
        "n_within_15pct": sum(r["rel_err_pct"] <= 15 for r in rows7),
    },
    "sec7_distribution_tests": {
        "speed_pdf_rmse_raw_bank": round(speed_rmse_raw, 5),
        "speed_pdf_rmse_microtrip_bank": round(speed_rmse_mt, 5),
        "va_pdf_rmse_raw_bank": round(va_rmse_raw, 5),
        "va_pdf_rmse_microtrip_bank": round(va_rmse_mt, 5),
        "ks_speed_raw": {"D": round(float(ks_v_raw.statistic), 4),
                         "p": float(ks_v_raw.pvalue)},
        "ks_speed_microtrip": {"D": round(float(ks_v_mt.statistic), 4),
                               "p": float(ks_v_mt.pvalue)},
        "ks_accel_microtrip": {"D": round(float(ks_a_mt.statistic), 4),
                               "p": float(ks_a_mt.pvalue)},
        "note": ("KS p-values are dominated by the 7.0M-sample fleet size; "
                 "the D statistic (max CDF deviation) is the interpretable "
                 "effect size. 'raw bank' = pooled 1 Hz traces including "
                 "creep segments excluded by the micro-trip filter; "
                 "'micro-trip bank' = the construction frame."),
    },
    "sec8_route_split": {
        "train_trips": int(f_tr["trip_id"].nunique()),
        "test_trips": int(f_te["trip_id"].nunique()),
        "train_microtrips": int(len(f_tr)),
        "test_microtrips": int(len(f_te)),
        "tetdc1_vs_test": {"feature_rows": rows8_route,
                            "composite_err_pct": round(comp8_route, 2)},
        "cv_rebuild_vs_test": {"feature_rows": rows8_rebuild,
                               "composite_err_pct": round(comp8_rebuild, 2)},
        "bootstrap_n": n_boot,
        "bootstrap_composite_err_pct": boot_comp_s,
        "bootstrap_per_feature": boot_summary,
    },
    "sec8_time_split": {
        "peak_hours_utc": sorted(PEAK_HOURS),
        "peak_microtrips": int(len(f_pk)),
        "offpeak_microtrips": int(len(f_op)),
        "tetdc1_vs_peak": {"composite_err_pct": round(comp8_peak, 2),
                            "feature_rows": rows8_peak},
        "tetdc1_vs_offpeak": {"composite_err_pct": round(comp8_offpeak, 2),
                               "feature_rows": rows8_offpeak},
        "offpeak_rebuild_vs_peak": {
            "composite_err_pct": round(comp8_transfer, 2),
            "feature_rows": rows8_transfer},
    },
    "sec13_tehran_cycles_baseline": tehran_cycles,
}
out = os.path.join(OUTD, "validation_results.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1, ensure_ascii=False, cls=NpEncoder)

pd.DataFrame({"t_s": np.arange(len(v_cv)), "v_ms": v_cv,
              "v_kmh": np.round(v_cv * 3.6, 2)}
             ).to_csv(os.path.join(OUTD, "TETDC_CV_route_rebuild.csv"),
                      index=False)

print(json.dumps({
    "sec7_composite": round(comp7, 2),
    "sec7_within10": result["sec7_feature_validation"]["n_within_10pct"],
    "sec7_within15": result["sec7_feature_validation"]["n_within_15pct"],
    "sec7_ksD_mt": round(float(ks_v_mt.statistic), 4),
    "sec8_route_tetdc1_vs_test": round(comp8_route, 2),
    "sec8_route_rebuild_vs_test": round(comp8_rebuild, 2),
    "sec8_bootstrap_p50": boot_comp_s["p50"],
    "sec8_time_peak": round(comp8_peak, 2),
    "sec8_time_offpeak": round(comp8_offpeak, 2),
    "sec8_transfer_offpeak_to_peak": round(comp8_transfer, 2),
}, indent=1, cls=NpEncoder))
print("saved", out)
