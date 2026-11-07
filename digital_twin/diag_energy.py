"""Diagnostic: energy decomposition of TETDC-1 flat vs mixed-grade runs."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_model import EVDigitalTwin

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cyc = pd.read_csv(os.path.join(BASE, "data", "processed", "TETDC_1.csv"))
v = cyc["v_ms"].to_numpy()
T = len(v)
twin = EVDigitalTwin(season="summer")

# sanity 1: constant 7.66 m/s flat
r_const = twin.simulate(np.full(1800, 7.66), 0.0)
print(f"const 7.66 m/s flat: {r_const['kWh_per_100km']:.2f} kWh/100km")

# sanity 2: constant at cycle v_mean with grade mix
gseq = np.tile(np.repeat([0.0328, -0.0328], 300), int(np.ceil(T / 600)))[:T]
r_const_mix = twin.simulate(np.full(T, 7.66), gseq)
print(f"const 7.66 m/s mixed grade: {r_const_mix['kWh_per_100km']:.2f} kWh/100km")

# full cycle decomposition
for label, grade in [("flat", 0.0), ("mix", gseq)]:
    theta = np.arctan(grade if np.isscalar(grade) else grade)
    a = np.gradient(v, 1.0)
    F_aero = 0.5 * twin.rho * twin.Cd * twin.A * v**2
    F_roll = twin.m * 9.81 * twin.Cr * np.cos(theta)
    F_grade = twin.m * 9.81 * np.sin(theta)
    F_iner = twin.m * a
    P_wheel = (F_iner + F_aero + F_roll + F_grade) * v
    r = twin.simulate(v, grade)
    MJ = 1e-6
    dist = v.sum()
    print(f"\n== {label} == kWh/100km {r['kWh_per_100km']:.2f}")
    print(f"  E_wheel: aero {np.trapezoid(F_aero*v)*MJ:.3f} | roll {np.trapezoid(F_roll*v)*MJ:.3f} |"
          f" grade {np.trapezoid(F_grade*v)*MJ:.3f} | inertial {np.trapezoid(F_iner*v)*MJ:.3f} (MJ)")
    print(f"  P_wheel>0 traction work {np.trapezoid(np.where(P_wheel>0,P_wheel,0))*MJ:.3f} MJ |"
          f" braking avail {np.trapezoid(np.where(P_wheel<0,-P_wheel,0))*MJ:.3f} MJ")
    print(f"  aux {twin.P_aux*T*MJ:.3f} MJ | E_term {r['E_batt_MJ']:.3f} MJ")
    # how much braking is at v < 0.5 (stop idle: no regen possible)?
    brk = np.where(P_wheel < 0, -P_wheel, 0)
    print(f"  braking while v>=0.5: {np.trapezoid(brk*(v>=0.5))*MJ:.3f} MJ |"
          f" lost at v<0.5: {np.trapezoid(brk*(v<0.5))*MJ:.3f} MJ")
