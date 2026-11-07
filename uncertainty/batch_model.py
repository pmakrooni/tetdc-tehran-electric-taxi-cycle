"""Vectorised re-implementation of tetdc/digital_twin/ev_model.py (EVDigitalTwin.simulate).
Identical equations; evaluates N parameter sets at once. Verified against the original."""
import numpy as np
G = 9.81
def simulate_batch(v, grade, prm, soc0=1.0, dt=1.0):
    """v: (T,) m/s; grade: (N,T) or (N,1) fraction; prm: dict of (N,1) arrays.
    returns E_batt_kWh (N,), distance_m (scalar)"""
    v = np.asarray(v, float)[None, :]
    theta = np.arctan(grade)
    a = np.gradient(v[0], dt)[None, :]
    m = prm['m']; F_aero = 0.5 * prm['rho'] * prm['Cd'] * prm['A'] * v**2
    F_tract = m*a + F_aero + m*G*prm['Cr']*np.cos(theta) + m*G*np.sin(theta)
    P_wheel = F_tract * v
    T_wheel = F_tract * prm['rw']
    T_m = np.abs(T_wheel) / prm['gr']
    T_hat = np.clip(T_m / (prm['P_motor_max'] / prm['gr'] / 2.0), 0.02, 1.6)
    eta_m = np.clip(prm['eta_peak'] - 0.28*(T_hat-0.75)**2, 0.62, prm['eta_peak'])
    eta_i = 0.93 + 0.04*(1 - np.exp(-8.0*T_hat))
    P_trac = np.where(P_wheel > 0, P_wheel/(eta_m*eta_i), 0.0)
    P_trac = np.minimum(P_trac, prm['P_motor_max'])
    P_regen_cap = np.maximum(P_wheel, -prm['P_regen_max'])
    P_brg = np.where(P_wheel < 0, P_regen_cap*prm['eta_regen'], 0.0)
    P_elec = P_trac + P_brg + prm['P_aux']
    N, T = P_elec.shape
    soc = np.full(N, soc0); R = prm['R_int'][:, 0]; Euse = prm['E_use'][:, 0]
    P_term = np.empty((N, T))
    for k in range(T):
        V = 300.0 + 110.0*np.clip(soc, 0, 1)
        Pr = P_elec[:, k]
        disc = np.sqrt(np.maximum(V**2 - 4*R*Pr, 0.0))
        I = (V - disc)/(2*R)
        I = np.where(Pr > 0, np.minimum(I, 600.0), np.maximum(I, -450.0))
        Pk = V*I - I**2*R
        P_term[:, k] = Pk
        soc = np.clip(soc - Pk*dt/Euse, 0.0, 1.0)
    dist = float(np.cumsum(v[0])[-1]*dt)
    E_kWh = np.trapezoid(P_term, dx=dt, axis=1)/1e6/3.6
    return E_kWh, dist
