"""TETDC Stage 6b — drive-cycle construction: Markov MC vs micro-trip
recombination, best-of selection (per prompt section 4: 'Select the method
based on data characteristics').

Method A: kinematic Markov chain (33 speed states @ 1 Hz), Monte Carlo
          synthesis, 8 candidates.
Method B: recombination of REAL 1 Hz micro-trip traces from the 9,000-trip
          sample (random-search assembly, best of 1,500; preserves true
          v_max / a_max / d_max kinematics by construction).
Both scored on identical duration-weighted fleet targets; extremes (v_max,
a_max, d_max) target the 1 Hz p99/p99/p01 of the trace sample.
Output: TETDC_1.csv, markov_matrix.npz, drive_cycle_stats.json
"""
import json, os
import numpy as np
import pandas as pd

SEED = int(os.environ.get("TETDC_SEED", 42))   # override only for the seed-variability check
rng = np.random.default_rng(SEED)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.path.join(BASE, "data", "processed")
OUTW = os.environ.get("TETDC_OUTDIR", OUTD)    # outputs; default = data/processed
os.makedirs(OUTW, exist_ok=True)
TRACES = os.path.join(OUTD, "microtrips_1hz_sample.parquet")
FEATS = os.path.join(OUTD, "microtrips_features.parquet")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_microtrips import microtrips  # same segmentation rule

V_BINS = np.arange(0, 34.0, 1.0)
NB = len(V_BINS) - 1
T_TARGET, T_LO, T_HI = 1800, 1700, 1900

def bin_of(v):
    b = np.clip(np.searchsorted(V_BINS, np.asarray(v), side="right") - 1, 0, NB - 1)
    return b.astype(np.int64) if np.ndim(b) else int(b)

# ---------- 1) load traces, build Markov matrix + micro-trip trace bank ------
tr = pd.read_parquet(TRACES)
N = np.zeros((NB, NB), dtype=np.int64)
mt_traces, mt_feats = [], []
all_v, all_acc = [], []
for (_, g) in tr.groupby("trip_id", sort=False):
    v = g["v_ms"].to_numpy()
    if len(v) < 2:
        continue
    i, j = bin_of(v[:-1]), bin_of(v[1:])
    np.add.at(N, (i, j), 1)
    all_v.append(v)
    all_acc.append(np.gradient(v, 1.0))
    for (_, _, seg) in microtrips(v):
        mt_traces.append(seg)
V_cat = np.concatenate(all_v); A_cat = np.concatenate(all_acc)
P = N / np.maximum(N.sum(axis=1, keepdims=True), 1)
empty = N.sum(axis=1) == 0
P[empty, :] = 0.0; P[empty, empty] = 1.0
np.savez_compressed(os.path.join(OUTW, "markov_matrix.npz"), N=N, P=P, bins=V_BINS)
print(f"Markov {NB}x{NB} | visited {(~empty).sum()} | transitions {int(N.sum()):,}"
      f" | microtrip traces {len(mt_traces):,}")

# ---------- 2) fleet targets (duration-weighted + 1 Hz extremes) -------------
feats = pd.read_parquet(FEATS)
w = feats["duration_s"]
target = {
    "v_mean": float((feats["v_mean"] * w).sum() / w.sum()),
    "P_idle": float((feats["P_idle"] * w).sum() / w.sum()),
    "a_rms": float((feats["a_rms"] * w).sum() / w.sum()),
    "v_max": float(np.percentile(V_cat, 99)),
    "a_max": float(np.percentile(A_cat, 99)),
    "d_max": float(np.percentile(A_cat, 1)),
}
del V_cat, A_cat, all_v, all_acc
print("fleet targets:", json.dumps(target, indent=1))

def cycle_features(v):
    acc = np.gradient(v, 1.0)
    return {"v_mean": float(v.mean()), "v_max": float(v.max()),
            "a_max": float(acc.max()), "d_max": float(acc.min()),
            "P_idle": float((v < 0.5).mean() * 100),
            "a_rms": float(np.sqrt((acc ** 2).mean())),
            "duration_s": int(len(v)), "distance_m": float(v.sum())}

KEYS = ["v_mean", "v_max", "a_max", "d_max", "P_idle", "a_rms"]
def composite_err(f):
    e = []
    for k in KEYS:
        t = abs(target[k]) if k == "d_max" else target[k]
        c = abs(f[k]) if k == "d_max" else f[k]
        e.append(abs(c - t) / max(abs(t), 1e-6))
    return float(np.mean(e))

# ---------- 3A) Method A: Markov MC synthesis --------------------------------
def synth_mc(seed, oversample=2400):
    r = np.random.default_rng(seed)
    v = np.empty(oversample); v[0] = 0.0; cur = 0
    for t in range(1, oversample):
        nxt = int(r.choice(NB, p=P[cur]))
        frac = v[t - 1] - np.floor(v[t - 1])
        v[t] = np.clip(V_BINS[nxt] + frac, 0.0, 33.0)
        cur = nxt
    v[1:-1] = np.median(np.stack([v[:-2], v[1:-1], v[2:]]), axis=0)
    stops = np.where((v < 0.5) & (np.arange(len(v)) >= int(0.85 * T_TARGET)))[0]
    end = int(stops[0]) if len(stops) else oversample
    start = 0
    while start < end and v[start] < 0.5:
        start += 1
    start = max(start - 30, 0)
    return np.round(v[start:end + 1], 3)

cands_a = []
for i in range(8):
    sd = SEED + 100 * (i + 1)
    v = synth_mc(sd); f = cycle_features(v)
    cands_a.append({"method": "markov_mc", "seed": sd, "features": f,
                    "composite_err": composite_err(f), "trace": v})
best_a = min(cands_a, key=lambda c: c["composite_err"])
print(f"[A markov_mc] best seed {best_a['seed']} err {best_a['composite_err']*100:.1f}% "
      f"(v_mean {best_a['features']['v_mean']:.2f}, v_max {best_a['features']['v_max']:.1f})")

# ---------- 3B) Method B: real micro-trip recombination ----------------------
mt_lens = np.array([len(s) for s in mt_traces])
r2 = np.random.default_rng(SEED + 7)
best_b = None
N_TRY = 1500
for it in range(N_TRY):
    parts, tot = [], 0
    while tot < T_LO:
        k = int(r2.integers(len(mt_traces)))
        parts.append(mt_traces[k]); tot += mt_lens[k]
        if tot > 1.15 * T_HI:            # too long: reject assembly early
            break
    if not (T_LO <= tot <= T_HI + 60):
        continue
    v = np.concatenate(parts)
    f = cycle_features(v)
    err = composite_err(f)
    if best_b is None or err < best_b["composite_err"]:
        best_b = {"method": "microtrip_recomb", "seed": int(SEED + 7),
                  "iter": it, "features": f, "composite_err": err,
                  "trace": np.round(v, 3), "n_parts": len(parts)}
print(f"[B recombination] random-search err {best_b['composite_err']*100:.1f}% "
      f"(v_mean {best_b['features']['v_mean']:.2f}, v_max {best_b['features']['v_max']:.1f}, "
      f"parts {best_b['n_parts']})")

# local search (hill-climbing): swap one part at a time to reduce error
parts = list(np.where([True])[0])          # placeholder, replaced below
# recover the part indices of best_b by re-assembly is costly; instead hill-climb
# on a copy: we rebuild from stored best trace by matching exact segments.
# Simpler robust approach: independent hill-climbing starting from best_b's trace
# is index-less; so we re-run assembly WITH part indices tracked.
idx_parts, tot, tries = [], 0, 0
while tot < T_LO and tries < 500:
    k = int(r2.integers(len(mt_traces)))
    if tot + mt_lens[k] <= T_HI + 60:
        idx_parts.append(k); tot += mt_lens[k]
    tries += 1
cur_v = np.concatenate([mt_traces[k] for k in idx_parts])
cur_f = cycle_features(cur_v); cur_err = composite_err(cur_f)
improved = 0
for it in range(6000):
    if len(idx_parts) == 0:
        break
    pos = int(r2.integers(len(idx_parts)))
    k = int(r2.integers(len(mt_traces)))
    new_tot = tot - mt_lens[idx_parts[pos]] + mt_lens[k]
    if not (T_LO <= new_tot <= T_HI + 60):
        continue
    trial = idx_parts.copy(); trial[pos] = k
    v = np.concatenate([mt_traces[j] for j in trial])
    f = cycle_features(v); e = composite_err(f)
    if e < cur_err:
        idx_parts, tot, cur_v, cur_f, cur_err = trial, new_tot, v, f, e
        improved += 1
print(f"[B hill-climb] {improved} accepted swaps -> err {cur_err*100:.1f}%")
if cur_err < best_b["composite_err"]:
    best_b = {"method": "microtrip_recomb", "seed": int(SEED + 7),
              "iter": -improved, "features": cur_f, "composite_err": cur_err,
              "trace": np.round(cur_v, 3), "n_parts": len(idx_parts)}
print(f"[B final] err {best_b['composite_err']*100:.1f}% "
      f"(v_mean {best_b['features']['v_mean']:.2f}, v_max {best_b['features']['v_max']:.1f}, "
      f"parts {best_b['n_parts']})")

# ---------- 4) select overall best -> TETDC-1 ---------------------------------
final = min([best_a, best_b], key=lambda c: c["composite_err"])
bf = final["features"]
comp = []
for k in KEYS:
    t = abs(target[k]) if k == "d_max" else target[k]
    c = abs(bf[k]) if k == "d_max" else bf[k]
    comp.append({"feature": k, "target": round(t, 3), "cycle": round(c, 3),
                 "rel_err_pct": round(100 * abs(c - t) / max(abs(t), 1e-6), 1)})
print(f"TETDC-1 <- {final['method']} (err {final['composite_err']*100:.1f}%)")
for c in comp:
    print("  ", c)

pd.DataFrame({"t_s": np.arange(len(final["trace"])),
              "v_ms": final["trace"],
              "v_kmh": np.round(final["trace"] * 3.6, 2)}
             ).to_csv(os.path.join(OUTW, "TETDC_1.csv"), index=False)

stats = {
    "seed": SEED,
    "markov": {"states": NB, "visited_states": int((~empty).sum()),
               "transitions": int(N.sum()), "matrix": "markov_matrix.npz"},
    "fleet_targets": {k: round(x, 3) for k, x in target.items()},
    "method_A_markov_mc": {"best_seed": best_a["seed"],
        "composite_err_pct": round(best_a["composite_err"] * 100, 2),
        "features": {k: round(v, 3) for k, v in best_a["features"].items()}},
    "method_B_microtrip_recombination": {"iterations": N_TRY,
        "composite_err_pct": round(best_b["composite_err"] * 100, 2),
        "n_microtrips_used": int(best_b["n_parts"]),
        "features": {k: round(v, 3) for k, v in best_b["features"].items()}},
    "selected_method": final["method"],
    "selected": {"features": {k: (round(x, 3) if isinstance(x, float) else x)
                              for k, x in bf.items()},
                 "composite_err_pct": round(final["composite_err"] * 100, 2)},
    "feature_comparison": comp,
    "cycle_length_s": int(len(final["trace"])),
    "cycle_distance_km": round(bf["distance_m"] / 1000, 2),
}
with open(os.path.join(OUTW, "drive_cycle_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=1, ensure_ascii=False)
print("saved TETDC_1.csv + drive_cycle_stats.json")
