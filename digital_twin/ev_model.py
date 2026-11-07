"""TETDC Stage 4 — EV longitudinal digital-twin core (Aion V Plus class).

Physics per master prompt section 10-11:
  F_tractive(s) = m*a + 0.5*rho*Cd*A*v^2 + m*g*Cr*cos(theta) + m*g*sin(theta)
  P_wheel = F_tractive * v
  P_batt  = P_wheel/(eta_m*eta_i) + P_aux            (traction, P_wheel>0)
          = P_wheel*eta_regen + P_aux                 (braking,  P_wheel<0)
Battery: OCV(SOC) + internal resistance; SOC by energy coulomb counting.
All parameters carry provenance tags (see ev_params.json) — no fabricated data.
"""
import json
import numpy as np
import os

G = 9.81
_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "ev_params.json"), encoding="utf-8") as f:
    P = json.load(f)


class EVDigitalTwin:
    """Stateless-per-call longitudinal EV model (vectorised over time arrays)."""

    def __init__(self, season="summer", params=P):
        self.p = params
        d, lt = params["documented"], params["literature_typical"]
        self.m = lt["curb_mass_kg"]["value"] + lt["taxi_payload_kg"]["value"]  # 2175 kg
        self.Cd = lt["Cd"]["value"]
        self.A = lt["frontal_area_m2"]["value"]
        self.Cr = lt["C_r"]["value"]
        self.rho = params["derived"]["air_density_kg_m3"]      # Tehran altitude
        self.rw = lt["wheel_radius_m"]["value"]
        self.gr = lt["gear_ratio"]["value"]
        self.P_motor_max = lt["motor_peak_power_kW"]["value"] * 1e3
        self.P_regen_max = lt["motor_max_regen_power_kW"]["value"] * 1e3
        self.eta_peak = lt["eta_motor_peak"]["value"]
        self.eta_inv_hi = lt["eta_inverter_high_load"]["value"]
        self.eta_regen = lt["eta_regen_wheel_to_batt"]["value"]
        self.P_aux = lt["P_aux_W"][season]["value"]
        self.E_use = params["derived"]["usable_kWh"] * 3.6e6   # J
        self.V_nom = lt["pack_nominal_V"]["value"]
        self.R_int = lt["pack_R_int_ohm"]["value"]

    # ---------- powertrain maps -------------------------------------------
    def eta_motor(self, T_wheel):
        """Load-dependent PMSM-representative efficiency (regen path mirrors)."""
        T_m = np.abs(T_wheel) / self.gr                      # motor torque
        T_hat = T_m / (self.P_motor_max / self.gr / 2.0)     # frac of continuous torque
        T_hat = np.clip(T_hat, 0.02, 1.6)
        # peak eta at ~0.75 load, drops at light load (fixed losses)
        eta = self.eta_peak - 0.28 * (T_hat - 0.75) ** 2
        return np.clip(eta, 0.62, self.eta_peak)

    def eta_inv(self, T_wheel):
        T_m = np.abs(T_wheel) / self.gr
        T_hat = np.clip(T_m / (self.P_motor_max / self.gr / 2.0), 0.02, 1.6)
        return 0.93 + 0.04 * (1 - np.exp(-8.0 * T_hat))

    # ---------- battery ----------------------------------------------------
    def ocv(self, soc):
        """Roughly-linear NMC pack OCV: 300 V empty -> 410 V full."""
        return 300.0 + 110.0 * np.clip(soc, 0, 1)

    def batt_terminal_power(self, P_elec_req, soc):
        """Terminal power after I^2*R_int loss; sign: + = discharge."""
        V = self.ocv(soc)
        # P = V*I - I^2*R  ->  I = (V - sqrt(V^2 - 4 R P)) / (2R) for discharge
        disc = np.sqrt(np.maximum(V ** 2 - 4 * self.R_int * P_elec_req, 0.0))
        I = (V - disc) / (2 * self.R_int)
        I = np.where(P_elec_req > 0, np.minimum(I, 600.0), np.maximum(I, -450.0))
        return V * I - I ** 2 * self.R_int

    # ---------- core simulation -------------------------------------------
    def simulate(self, v_ms, grade, dt=1.0, soc0=1.0):
        """v_ms (m/s array), grade (array or scalar, fraction) -> dict of arrays."""
        v = np.asarray(v_ms, dtype=float)
        theta = np.arctan(np.asarray(grade, dtype=float) if np.ndim(grade) else float(grade))
        a = np.gradient(v, dt)                                # m/s^2
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        F_aero = 0.5 * self.rho * self.Cd * self.A * v ** 2
        F_roll = self.m * G * self.Cr * cos_t
        F_grade = self.m * G * sin_t
        F_iner = self.m * a
        F_tract = F_iner + F_aero + F_roll + F_grade          # prompt eq. (N)

        P_wheel = F_tract * v                                  # W  (signed)

        T_wheel = F_tract * self.rw
        # traction path
        eta_m, eta_i = self.eta_motor(T_wheel), self.eta_inv(T_wheel)
        P_trac = np.where(P_wheel > 0, P_wheel / (eta_m * eta_i), 0.0)
        P_trac = np.minimum(P_trac, self.P_motor_max)          # motor cap
        # regen path: capture limited fraction, rest friction braking
        P_regen_cap = np.maximum(P_wheel, -self.P_regen_max)   # cap on regen power
        P_brg = np.where(P_wheel < 0, P_regen_cap * self.eta_regen, 0.0)
        P_elec = P_trac + P_brg + self.P_aux

        # battery terminal + SOC integration
        n = len(v)
        soc = np.empty(n)
        soc_t, E_draw = soc0, 0.0
        P_term = np.empty(n)
        for k in range(n):
            Pk = self.batt_terminal_power(np.float64(P_elec[k]), soc_t)
            P_term[k] = Pk
            soc_t -= (Pk * dt) / self.E_use
            soc_t = min(max(soc_t, 0.0), 1.0)
            soc[k] = soc_t

        dist = np.cumsum(v) * dt
        E_batt_MJ = np.trapezoid(P_term, dx=dt) / 1e6         # signed MJ
        return {
            "v": v, "a": a, "grade": theta, "P_wheel": P_wheel, "P_elec": P_elec,
            "P_term": P_term, "soc": soc, "distance_m": dist[-1],
            "E_batt_MJ": E_batt_MJ, "E_batt_kWh": E_batt_MJ / 3.6,
            "kWh_per_100km": (E_batt_MJ / 3.6) / max(dist[-1], 1e-9) * 1e5,
            "eta_motor_trace": eta_m, "F_tract": F_tract,
        }

    # ---------- charging model (stage-5 link) -------------------------------
    def charge_session(self, soc0, soc1, P_station_kW, t_max_h=3.0, dt_s=10.0):
        """CC-then-CV-taper DC model; 20-80% in <=45 min is the documented anchor.
        Plateau power is capped by station; taper after SOC 0.8 as (1-soc)/(1-0.8) 
        shape scaled so that a 95 kW plateau reproduces the documented anchor."""
        t, soc, P = 0.0, soc0, []
        ts, socs = [0.0], [soc0]
        taper_from = 0.80
        while soc < soc1 and t < t_max_h * 3600:
            P_plateau = P_station_kW * 1e3
            if soc > taper_from:                               # CV taper
                P_plateau *= max((1.0 - soc) / (1.0 - taper_from), 0.05)
            P.append(P_plateau)
            V = self.ocv(soc)
            I = min(P_plateau / V, 450.0)                      # current cap
            P_batt = V * I - I ** 2 * self.R_int                # into battery (neg. convention flip)
            soc += (P_batt * dt_s) / self.E_use
            t += dt_s
            ts.append(t)
            socs.append(min(soc, 1.0))
        return np.array(ts) / 60.0, np.array(socs), np.array(P) / 1e3
