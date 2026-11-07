"""TETDC Stage 7 — Fleet-level digital-twin simulation (24 h, event-based).

Simulates the CURRENT Tehran electric-taxi fleet (~470 circulating EVs,
feasibility report anchor) over one representative weekday:
  - trip chains: Poisson-thinning arrivals, hourly shape + distance/speed
    distributions transferred from Porto (stage-3 chain params, calibrated
    to Tehran aggregate 43.75 trips/taxi/day)
  - trip energy: digital-twin kWh/100km city-mixed rate (stage-6d S3) with
    per-trip route heterogeneity (lognormal, CV 18%)
  - charging: 19 spatial stations (stage-5 layer), FIFO plug queues,
    CC-CV sessions from the validated twin charge model
KPIs: fleet energy, charging-demand profile (MW), plug utilization,
queue waits, range-anxiety events, SOC trajectories.

Integrity note (§32): no Tehran GPS data exists publicly — pattern transfer
is documented in tehran_chain_params.json; every assumption below is tagged.
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "digital_twin"))
from ev_model import EVDigitalTwin

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(PROC, "fleet_sim_results.json")

SEED = 42
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ inputs
chain = json.load(open(os.path.join(PROC, "tehran_chain_params.json"), encoding="utf-8"))
hourly = np.array([chain["porto_pattern_transfer"]["hourly_start_share_weekday"][str(h)] for h in range(24)])
trips_per_taxi_day = chain["tehran_official_anchors"]["trips_per_taxi_per_day"]  # 43.75
N_EV = 470                       # current circulating EV taxis (feasibility report)
SEASON = "summer"                # worst-case aux load season
E_USABLE = 85.6                  # kWh (ev_params derived)
CITY_RATE = json.load(open(os.path.join(PROC, "cycle_energy_results.json"), encoding="utf-8"))\
    ["season_summer"]["S3_roundtrip_mix"]["kWh_per_100km"] / 100.0     # kWh/km

# charge policy
SOC_CHARGE_TH = 0.25     # seek station below this SOC
SOC_TARGET = 0.80         # opportunity-charge ceiling (battery-health practice)
SOC_STRAND = 0.05         # range-anxiety threshold

# ------------------------------------------------- charging station layer
cs = pd.read_csv(os.path.join(PROC, "charging_stations.csv"))
cs_spatial = cs.dropna(subset=["lat", "lon"]).copy()
# de-duplicate documented entries that share a location with OSM ones (<150 m)
used, stations = [], []
for _, r in cs_spatial.sort_values(["entry_type"]).iterrows():
    dup = any((abs(r.lat - u.lat) < 0.0015) and (abs(r.lon - u.lon) < 0.0015) for u in used)
    if not dup:
        used.append(r)
        stations.append(r)
stations = pd.DataFrame(used).reset_index(drop=True)

def plug_config(row):
    """(n_plugs, plug_kW, tag) — documented power -> plugs; else literature-typical."""
    p = row.get("power_kW")
    if np.isfinite(p) and p >= 1000:      # Mapna plaza hub 5000 kW
        return 50, 100.0, "documented-plaza->50x100kW"
    if np.isfinite(p) and p >= 300:       # Karamid 340 kW site
        return 4, 85.0, "documented-site->4x85kW"
    if np.isfinite(p) and p >= 100:       # 120 kW site
        return 2, 60.0, "documented-site->2x60kW"
    return 2, 90.0, "literature-typical 2x90kW dual dispenser"

n_plugs = np.array([plug_config(r)[0] for _, r in stations.iterrows()], dtype=int)
plug_kW = np.array([plug_config(r)[1] for _, r in stations.iterrows()], dtype=float)
plug_tags = [plug_config(r)[2] for _, r in stations.iterrows()]
N_ST = len(stations)

# taxi home lat-band -> nearest station index (band 0..9 over lat range)
LAT_LO, LAT_HI = 35.60, 35.85
st_lat = stations["lat"].to_numpy()
def nearest_station(lat):
    return int(np.argmin(np.abs(st_lat - lat)))

# ------------------------------------------------------------- twin + rate
twin = EVDigitalTwin(season=SEASON)
# distance ~ LogNormal fitted to Porto percentiles (median 4.03, p90 11.04)
MU_D, SIG_D = np.log(4.03), (np.log(11.04) - np.log(4.03)) / 1.2816
# avg speed ~ LogNormal (median 23.8, p90 44.5 km/h)
MU_S, SIG_S = np.log(23.8), (np.log(44.5) - np.log(23.8)) / 1.2816
# per-trip route heterogeneity around city-mixed rate
SIG_R = 0.18

# --------------------------------------------------------- trip generation
# per taxi: number of trips ~ Poisson(43.75); start times ~ hourly profile
# TWO simulated days: day 1 = warm-up, day 2 = steady-state reporting
N_DAYS = 2

def gen_trips():
    t0s, durs, es, ds = [], [], [], []
    for day in range(N_DAYS):
        n = rng.poisson(trips_per_taxi_day)
        hours = rng.choice(24, size=n, p=hourly / hourly.sum())
        start_min = hours * 60 + rng.integers(0, 60, size=n)
        d = np.minimum(rng.lognormal(MU_D, SIG_D, n), 35.0)              # km, metro cap
        s = np.clip(rng.lognormal(MU_S, SIG_S, n), 8.0, 60.0)           # km/h
        t0s.append(start_min + day * 1440)
        durs.append(d / s * 60.0)                                       # min
        ds.append(d)
        es.append(d * CITY_RATE * np.clip(rng.lognormal(0.0, SIG_R, n), 0.65, 1.9))
    t0 = np.concatenate(t0s).astype(float)
    order = np.argsort(t0)
    return dict(t0=t0[order], dur=np.concatenate(durs)[order],
                e=np.concatenate(es)[order], d=np.concatenate(ds)[order])

# charge-time lookup: soc0 -> (minutes, mean kW) per plug power, target 0.8
PLUG_POWERS = sorted(set(plug_kW.tolist()))
lookup = {}
for pw in PLUG_POWERS:
    rows = []
    for s0 in np.arange(0.05, 0.751, 0.025):
        t, socs, P = twin.charge_session(float(s0), SOC_TARGET, pw, t_max_h=3.0)
        rows.append((s0, t[-1], float(np.mean(P))))
    lookup[pw] = np.array(rows)

def charge_time_min(soc0, pw):
    tab = lookup[pw]
    return float(np.interp(soc0, tab[:, 0], tab[:, 1])), \
           float(np.interp(soc0, tab[:, 0], tab[:, 2]))

# ----------------------------------------------------------------- sim loop
DT = 1.0                                    # minute resolution
T_END = 1440 * N_DAYS
soc = np.full(N_EV, rng.uniform(0.55, 0.95, N_EV))     # day-0 arbitrary SOC (warm-up)
home_lat = rng.uniform(LAT_LO + 0.05, LAT_HI - 0.05, N_EV)
home_st = np.array([nearest_station(la) for la in home_lat])
trips = [gen_trips() for _ in range(N_EV)]
ti = np.zeros(N_EV, dtype=int)              # next trip index per taxi
WARM = 1440                                 # day-1 warm-up cut
ti_snap = np.zeros(N_EV, dtype=int)         # completions at warm-up cut
sess_snap = 0                               # sessions at warm-up cut

state = np.array(["idle"] * N_EV, dtype=object)   # idle | trip | queue | charge
t_end_trip = np.full(N_EV, np.inf)
t_end_charge = np.full(N_EV, np.inf)
cur_st = np.full(N_EV, -1)                  # station occupied/queued at
strand_events, queue_wait_log = [], []

plug_busy = [np.zeros(n, dtype=bool) for n in n_plugs]      # per station
plug_taxi = [[-1] * n for n in n_plugs]
queue = [[] for _ in range(N_ST)]                           # FIFO taxi ids
queue_enter = np.full(N_EV, np.nan)                         # minute tax i entered queue
sess_pw = np.zeros(N_EV)                                     # mean session power (kW)
n_sessions = 0

mw_profile = np.zeros(T_END)                 # aggregate charging power (MW)
soc_hist = np.zeros((T_END, N_EV))
stranded_flag = np.zeros(N_EV, dtype=bool)

def free_plug(s):
    for i in range(n_plugs[s]):
        if not plug_busy[s][i]:
            return i
    return -1

def start_charge(i, s, m):
    """assign taxi i to a free plug at station s at minute m."""
    global n_sessions
    p = free_plug(s)
    if p < 0:
        return False
    t_c, pw_mean = charge_time_min(soc[i], plug_kW[s])
    plug_busy[s][p] = True; plug_taxi[s][p] = i
    t_end_charge[i] = m + t_c
    state[i], cur_st[i] = "charge", s
    sess_pw[i] = pw_mean
    n_sessions += 1
    return True

def end_charge(i, m):
    """release plug, set SOC=target, pop queue."""
    s = cur_st[i]
    for k in range(len(plug_taxi[s])):
        if plug_taxi[s][k] == i:
            plug_taxi[s][k] = -1
            plug_busy[s][k] = False
            break
    soc[i] = SOC_TARGET
    cur_st[i], state[i] = -1, "idle"
    if queue[s]:
        j = queue[s].pop(0)
        start_charge(j, s, m)
        queue_wait_log.append(m - queue_enter[j]) if np.isfinite(queue_enter[j]) else queue_wait_log.append(0.0)

for minute in range(T_END):
    if minute == WARM:
        ti_snap[:] = ti
        sess_snap = n_sessions
        queue_wait_log.clear()
        strand_events.clear()
        stranded_flag[:] = False
    # 1) finish charging -> SOC = target, free plug, pop queue
    for i in range(N_EV):
        if state[i] == "charge" and minute >= t_end_charge[i]:
            end_charge(i, minute)
    # 2) finish trips -> deduct energy, maybe seek charge
    for i in range(N_EV):
        if state[i] == "trip" and minute >= t_end_trip[i]:
            dE = trips[i]["e"][ti[i] - 1]
            soc[i] -= dE / E_USABLE
            if soc[i] < SOC_STRAND:
                strand_events.append(dict(taxi=i, minute=minute,
                                          soc=round(float(soc[i]), 3)))
                soc[i] = max(soc[i], 0.02)
                stranded_flag[i] = True
            state[i] = "idle"
            if soc[i] < SOC_CHARGE_TH:
                s = home_st[i]
                if not start_charge(i, s, minute):
                    queue[s].append(i)
                    queue_enter[i] = minute
                    state[i], cur_st[i] = "queue", s
    # 3) start new trips (or queue for charging if too low)
    for i in range(N_EV):
        if state[i] == "idle" and ti[i] < len(trips[i]["t0"]):
            if trips[i]["t0"][ti[i]] <= minute:
                if soc[i] > SOC_CHARGE_TH:
                    state[i] = "trip"
                    t_end_trip[i] = minute + trips[i]["dur"][ti[i]]
                    ti[i] += 1
                else:
                    s = home_st[i]
                    if state[i] != "queue":
                        if not start_charge(i, s, minute):
                            queue[s].append(i)
                            queue_enter[i] = minute
                            state[i], cur_st[i] = "queue", s
    # 4) queue -> plug when freed
    for s in range(N_ST):
        while queue[s]:
            if not start_charge(queue[s][0], s, minute):
                break
            j = queue[s].pop(0)
            queue_wait_log.append(minute - queue_enter[j]) if np.isfinite(queue_enter[j]) else queue_wait_log.append(0.0)
    # 5) aggregate charging power this minute (taxis currently charging)
    tot = 0.0
    for i in range(N_EV):
        if state[i] == "charge":
            tot += sess_pw[i]
    mw_profile[minute] = tot / 1000.0
    soc_hist[minute] = soc

mw2 = mw_profile[WARM:]                     # steady-state day
soc2 = soc_hist[WARM:]

# ------------------------------------------------------------------- KPIs
# day-2 (steady state) completions & energy
daily_e, daily_km, done2 = [], [], []
for i in range(N_EV):
    done = ti[i] - ti_snap[i]
    done2.append(done)
    daily_e.append(sum(trips[i]["e"][ti_snap[i]: ti_snap[i] + done]))
    daily_km.append(sum(trips[i]["d"][ti_snap[i]: ti_snap[i] + done]))
sessions = n_sessions - sess_snap
n_gen_day2 = sum(len(trips[i]["t0"]) - (ti_snap[i] if ti_snap[i] < len(trips[i]['t0']) else len(trips[i]['t0'])) for i in range(N_EV))
util = []
for s in range(N_ST):
    util.append({"station": str(stations.iloc[s]["station_id"]),
                 "name": str(stations.iloc[s].get("name", ""))[:40],
                 "n_plugs": int(n_plugs[s]), "plug_kW": float(plug_kW[s]),
                 "config_tag": plug_tags[s]})

out = {
    "purpose": "TETDC stage 7 — current-fleet 24h digital-twin simulation",
    "random_seed": SEED,
    "assumptions": [
        f"N_EV={N_EV} circulating electric taxis (feasibility-report anchor)",
        f"trip energy rate = {CITY_RATE*100:.2f} kWh/100km (stage-6d S3 city-mixed, summer) "
        f"with per-trip lognormal heterogeneity CV=18% (flat vs hilly route mix)",
        f"battery usable {E_USABLE} kWh; charge trigger SOC<{SOC_CHARGE_TH}, target {SOC_TARGET}",
        "plug configs: documented site powers converted to plugs; unknown-power stations "
        "assumed literature-typical dual 90 kW dispensers (flagged, NOT measured)",
        "minute-resolution event loop; CC-CV charge sessions from validated twin model",
        "taxi home lat-band -> nearest mapped station (no GPS positions exist; documented limitation)"
    ],
    "kpi": {
        "fleet_energy_MWh_day": round(float(np.sum(daily_e)) / 1000, 2),
        "energy_per_taxi_kWh_day_mean": round(float(np.mean(daily_e)), 2),
        "energy_per_taxi_kWh_day_std": round(float(np.std(daily_e)), 2),
        "km_per_taxi_day_mean": round(float(np.mean(daily_km)), 1),
        "trips_completed": int(sum(done2)),
        "trips_requested_day2": int(n_gen_day2),
        "service_loss_pct": round(100.0 * (1 - sum(done2) / max(n_gen_day2, 1)), 2),
        "charge_sessions": int(sessions),
        "sessions_per_taxi_day": round(float(sessions) / N_EV, 3),
        "queue_wait_min_median": round(float(np.median(queue_wait_log)), 1) if queue_wait_log else 0.0,
        "queue_wait_min_p90": round(float(np.percentile(queue_wait_log, 90)), 1) if queue_wait_log else 0.0,
        "charge_energy_MWh_day": round(float(mw2.sum()) / 60.0, 2),
        "range_anxiety_events": len(strand_events),
        "taxis_stranded_at_least_once": int(stranded_flag.sum()),
        "peak_charging_MW": round(float(mw2.max()), 3),
        "peak_hour": int(np.argmax(mw2)) // 60,
        "mean_charging_MW": round(float(mw2.mean()), 3),
    },
    "mw_profile_hourly": [round(float(mw2[h*60:(h+1)*60].mean()), 3) for h in range(24)],
    "soc_stats": {
        "end_of_day_mean": round(float(np.mean(soc2[-1])), 3),
        "min_soc_p5": round(float(np.percentile(soc2.min(axis=0), 5)), 3),
        "taxis_below_20pct_eod": int((soc2[-1] < 0.20).sum()),
    },
    "stations": util,
    "n_spatial_stations": int(N_ST),
    "n_plugs_total": int(n_plugs.sum()),
    "plug_power_MW_total": round(float((n_plugs * plug_kW).sum() / 1000), 1),
    "soc_trajectory_sample": {
        "taxis": [0, 1, 2, 3, 4, 5],
        "soc": {int(i): [round(float(x), 4) for x in soc2[::15, i]] for i in [0, 1, 2, 3, 4, 5]},
        "t_min": list(range(0, 1440, 15)),
    },
    "strand_events_sample": strand_events[:20],
    "n_days_simulated": N_DAYS,
    "reporting_day": "day 2 (steady state after warm-up)",
}
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT)
print(json.dumps(out["kpi"], indent=1))
print("plugs total:", out["n_plugs_total"], " MW:", out["plug_power_MW_total"])
