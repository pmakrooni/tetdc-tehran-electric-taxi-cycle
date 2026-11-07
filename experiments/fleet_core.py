"""TETDC Stage 9c core — vectorised fleet simulator (reusable for scenarios).

Semantics identical to experiments/run_fleet_sim.py (validated stage-7 run),
with integer state codes + numpy event scanning so 10k-taxi fleets run fast.
States: 0=idle 1=trip 2=queue 3=charge.
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "digital_twin"))
from ev_model import EVDigitalTwin

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")

IDLE, TRIP, QUEUE, CHARGE = 0, 1, 2, 3
LAT_LO, LAT_HI = 35.60, 35.85


def load_inputs():
    chain = json.load(open(os.path.join(PROC, "tehran_chain_params.json"), encoding="utf-8"))
    hourly = np.array([chain["porto_pattern_transfer"]["hourly_start_share_weekday"][str(h)] for h in range(24)])
    trips_per_taxi_day = chain["tehran_official_anchors"]["trips_per_taxi_per_day"]
    city_rate = json.load(open(os.path.join(PROC, "cycle_energy_results.json"), encoding="utf-8"))\
        ["season_summer"]["S3_roundtrip_mix"]["kWh_per_100km"] / 100.0
    cs = pd.read_csv(os.path.join(PROC, "charging_stations.csv"))
    cs_spatial = cs.dropna(subset=["lat", "lon"]).copy()
    used, stations = [], []
    for _, r in cs_spatial.sort_values(["entry_type"]).iterrows():
        if not any((abs(r.lat - u.lat) < 0.0015) and (abs(r.lon - u.lon) < 0.0015) for u in used):
            used.append(r)
            stations.append(r)
    stations = pd.DataFrame(used).reset_index(drop=True)

    def plug_config(row):
        p = row.get("power_kW")
        if np.isfinite(p) and p >= 1000:
            return 50, 100.0
        if np.isfinite(p) and p >= 300:
            return 4, 85.0
        if np.isfinite(p) and p >= 100:
            return 2, 60.0
        return 2, 90.0

    n_plugs = np.array([plug_config(r)[0] for _, r in stations.iterrows()], dtype=int)
    plug_kW = np.array([plug_config(r)[1] for _, r in stations.iterrows()], dtype=float)
    return hourly, trips_per_taxi_day, city_rate, stations, n_plugs, plug_kW


def run_fleet(N_EV=470, plug_mult=1.0, seed=42, season="summer",
              SOC_CHARGE_TH=0.25, SOC_TARGET=0.80, SOC_STRAND=0.05,
              E_USABLE=85.6, n_days=2, charge_lookup=None):
    """Returns KPI dict. plug_mult scales every station's plug count."""
    rng = np.random.default_rng(seed)
    hourly, tpd, city_rate, stations, n_plugs0, plug_kW = load_inputs()
    n_plugs = np.maximum(np.round(n_plugs0 * plug_mult).astype(int), 1)
    N_ST = len(stations)
    twin = EVDigitalTwin(season=season)
    if charge_lookup is None:
        charge_lookup = {}
        for pw in sorted(set(plug_kW.tolist())):
            rows = []
            for s0 in np.arange(0.05, 0.751, 0.025):
                t, socs, P = twin.charge_session(float(s0), SOC_TARGET, pw, t_max_h=3.0)
                rows.append((s0, t[-1], float(np.mean(P))))
            charge_lookup[pw] = np.array(rows)

    def charge_time_min(soc0, pw):
        tab = charge_lookup[pw]
        return float(np.interp(soc0, tab[:, 0], tab[:, 1])), \
               float(np.interp(soc0, tab[:, 0], tab[:, 2]))

    MU_D, SIG_D = np.log(4.03), (np.log(11.04) - np.log(4.03)) / 1.2816
    MU_S, SIG_S = np.log(23.8), (np.log(44.5) - np.log(23.8)) / 1.2816
    SIG_R = 0.18

    # ---- trips (n_days), concatenated per taxi
    t0_list, dur_list, e_list, d_list = [], [], [], []
    for i in range(N_EV):
        t0s, durs, es, ds = [], [], [], []
        for day in range(n_days):
            n = rng.poisson(tpd)
            hours = rng.choice(24, size=n, p=hourly / hourly.sum())
            start = hours * 60 + rng.integers(0, 60, size=n)
            d = np.minimum(rng.lognormal(MU_D, SIG_D, n), 35.0)
            s = np.clip(rng.lognormal(MU_S, SIG_S, n), 8.0, 60.0)
            t0s.append(start + day * 1440); durs.append(d / s * 60.0); ds.append(d)
            es.append(d * city_rate * np.clip(rng.lognormal(0, SIG_R, n), 0.65, 1.9))
        t0 = np.concatenate(t0s); order = np.argsort(t0)
        t0_list.append(t0[order]); dur_list.append(np.concatenate(durs)[order])
        e_list.append(np.concatenate(es)[order]); d_list.append(np.concatenate(ds)[order])

    soc = np.full(N_EV, rng.uniform(0.55, 0.95, N_EV))
    home_lat = rng.uniform(LAT_LO + 0.05, LAT_HI - 0.05, N_EV)
    st_lat = stations["lat"].to_numpy()
    home_st = np.abs(home_lat[:, None] - st_lat[None, :]).argmin(axis=1)

    state = np.zeros(N_EV, dtype=np.int8)          # 0 idle
    t_end_trip = np.full(N_EV, np.inf)
    t_end_charge = np.full(N_EV, np.inf)
    cur_st = np.full(N_EV, -1)
    queue_enter = np.full(N_EV, np.nan)
    sess_pw = np.zeros(N_EV)
    ti = np.zeros(N_EV, dtype=int)
    n_trips = np.array([len(t) for t in t0_list], dtype=int)
    next_t0 = np.array([t[0] if len(t) else np.inf for t in t0_list], dtype=float)  # next due trip
    n_sessions = 0
    strand_events = 0
    queue_waits = []

    plug_busy = [np.zeros(n, dtype=bool) for n in n_plugs]
    plug_taxi = [[-1] * n for n in n_plugs]
    queue = [[] for _ in range(N_ST)]

    def free_plug(s):
        for i in range(n_plugs[s]):
            if not plug_busy[s][i]:
                return i
        return -1

    def start_charge(i, s, m):
        nonlocal n_sessions
        p = free_plug(s)
        if p < 0:
            return False
        t_c, pw_mean = charge_time_min(soc[i], plug_kW[s])
        plug_busy[s][p] = True; plug_taxi[s][p] = i
        t_end_charge[i] = m + t_c
        state[i], cur_st[i] = CHARGE, s
        sess_pw[i] = pw_mean
        n_sessions += 1
        return True

    def end_charge(i, m):
        s = cur_st[i]
        for k in range(len(plug_taxi[s])):
            if plug_taxi[s][k] == i:
                plug_taxi[s][k] = -1
                plug_busy[s][k] = False
                break
        soc[i] = SOC_TARGET
        cur_st[i], state[i] = -1, IDLE
        if queue[s]:
            j = queue[s].pop(0)
            start_charge(j, s, m)
            queue_waits.append(m - queue_enter[j]) if np.isfinite(queue_enter[j]) else queue_waits.append(0.0)

    T_END = 1440 * n_days
    WARM = 1440
    mw_profile = np.zeros(T_END)
    ti_snap = np.zeros(N_EV, dtype=int)
    sess_snap = 0
    strand_day2 = 0
    E_day2 = np.zeros(N_EV)          # energy spent in day 2
    KM_day2 = np.zeros(N_EV)

    for minute in range(T_END):
        if minute == WARM:
            ti_snap[:] = ti
            sess_snap = n_sessions
            queue_waits.clear()
            strand_day2 = 0
        # 1) charge endings
        idx = np.where((state == CHARGE) & (minute >= t_end_charge))[0]
        for i in idx:
            end_charge(int(i), minute)
        # 2) trip endings
        idx = np.where((state == TRIP) & (minute >= t_end_trip))[0]
        for i in idx:
            i = int(i)
            k = ti[i] - 1
            dE = e_list[i][k]
            soc[i] -= dE / E_USABLE
            if minute >= WARM:
                E_day2[i] += dE
                KM_day2[i] += d_list[i][k]
            if soc[i] < SOC_STRAND:
                strand_events += 1
                if minute >= WARM:
                    strand_day2 += 1
                soc[i] = max(soc[i], 0.02)
            state[i] = IDLE
            if soc[i] < SOC_CHARGE_TH:
                s = home_st[i]
                if not start_charge(i, s, minute):
                    queue[s].append(i)
                    queue_enter[i] = minute
                    state[i], cur_st[i] = QUEUE, s
        # 3) departures
        due = np.where((state == IDLE) & (next_t0 <= minute))[0]
        for i in due:
            i = int(i)
            if soc[i] > SOC_CHARGE_TH:
                state[i] = TRIP
                t_end_trip[i] = minute + dur_list[i][ti[i]]
                ti[i] += 1
                next_t0[i] = t0_list[i][ti[i]] if ti[i] < n_trips[i] else np.inf
            elif state[i] != QUEUE:
                s = home_st[i]
                if not start_charge(i, s, minute):
                    queue[s].append(i)
                    queue_enter[i] = minute
                    state[i], cur_st[i] = QUEUE, s
        # 4) queue -> plugs
        for s in range(N_ST):
            while queue[s]:
                j = queue[s][0]
                if not start_charge(j, s, minute):
                    break
                queue[s].pop(0)
                queue_waits.append(minute - queue_enter[j]) if np.isfinite(queue_enter[j]) else queue_waits.append(0.0)
        # 5) aggregate MW
        mw_profile[minute] = sess_pw[state == CHARGE].sum() / 1000.0

    mw2 = mw_profile[WARM:]
    sessions2 = n_sessions - sess_snap
    trips_done2 = int(sum(ti[i] - ti_snap[i] for i in range(N_EV)))
    trips_req2 = int(sum(len(t0_list[i]) - ti_snap[i] for i in range(N_EV)))
    scale = N_EV / max(len(E_day2), 1)
    return {
        "N_EV": N_EV, "plug_mult": plug_mult, "seed": seed,
        "n_plugs_total": int(n_plugs.sum()),
        "plug_MW_total": round(float((n_plugs * plug_kW).sum() / 1000), 1),
        "kWh_per_taxi_day": round(float(E_day2.mean()), 2),
        "km_per_taxi_day": round(float(KM_day2.mean()), 1),
        "fleet_energy_MWh_day": round(float(E_day2.sum() / 1000), 2),
        "service_loss_pct": round(100.0 * (1 - trips_done2 / max(trips_req2, 1)), 2),
        "queue_wait_min_median": round(float(np.median(queue_waits)), 1) if queue_waits else 0.0,
        "queue_wait_min_p90": round(float(np.percentile(queue_waits, 90)), 1) if queue_waits else 0.0,
        "sessions_per_taxi_day": round(sessions2 / N_EV, 3),
        "peak_charging_MW": round(float(mw2.max()), 3),
        "mean_charging_MW": round(float(mw2.mean()), 3),
        "charge_energy_MWh_day": round(float(mw2.sum()) / 60.0, 2),
        "range_anxiety_events_day2": int(strand_day2),
        "eod_soc_mean": round(float(np.mean(soc)), 3),
    }
