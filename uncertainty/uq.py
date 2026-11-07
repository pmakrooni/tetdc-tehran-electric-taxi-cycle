"""Monte-Carlo and Sobol' uncertainty analysis of the TETDC energy comparison.
Usage: python uncertainty/uq.py            (all modes and seasons)
       python uncertainty/uq.py mc summer  (one mode/season)
15 uncertain inputs: 12 vehicle ranges from digital_twin/ev_params.json, road grade U(1.86%,3.28%), air density U(0.99,1.10),
and gross battery capacity U(62,80) kWh (published GAC Aion V Plus variants; ev_params.json's 93 kWh is the Aion LX Plus value).
Each draw is evaluated on TETDC-1 (flat and Tehran grade) and the official WLTC Class 3b (flat).
Modes mc_bc / sobol_bc add a 16th input, the energy bias of rebuilding 1 Hz speed from 15 s GPS positions,
sampled from U(0, 2b) with b from supplement/bias_factor.py; the flat TETDC-1 energy is multiplied by (1+bias)
and the grade effect is added unchanged."""
import sys, json, os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(ROOT,'digital_twin'))
OUTDIR=os.path.join(ROOT,'data','processed','uncertainty'); os.makedirs(OUTDIR,exist_ok=True)
from batch_model import simulate_batch
from ev_model import P as P0
from wltp.cycles import class3
from SALib.sample import sobol as sobol_sample
lt=P0['literature_typical']
vT=pd.read_csv(os.path.join(ROOT,'data','processed','TETDC_1.csv')).v_ms.values
vW=np.array(class3.class_data_b()['cycle'],float)/3.6
dT=np.cumsum(vT)
BIAS_FILE=next(f for f in [os.path.join(ROOT,'results','supplement','bias_factor.json'),
                            os.path.join(ROOT,'results_reference','supplement','bias_factor.json')] if os.path.exists(f))
def names_bounds(season,bc=False):
    nb=[('curb_mass_kg',lt['curb_mass_kg']['range']),('payload_kg',lt['taxi_payload_kg']['range']),
        ('Cd',lt['Cd']['range']),('A_m2',lt['frontal_area_m2']['range']),('Cr',lt['C_r']['range']),
        ('eta_motor_peak',lt['eta_motor_peak']['range']),('eta_regen',lt['eta_regen_wheel_to_batt']['range']),
        ('P_aux_W',lt['P_aux_W'][season]['range']),('R_int_ohm',lt['pack_R_int_ohm']['range']),
        ('P_regen_max_kW',lt['motor_max_regen_power_kW']['range']),('P_motor_max_kW',lt['motor_peak_power_kW']['range']),
        ('usable_frac',lt['battery_usable_fraction']['range']),('grade_frac',[0.0186,0.0328]),('rho',[0.99,1.10]),('battery_kWh',[62.0,80.0])]
    if bc: nb.append(('recon_bias',json.load(open(BIAS_FILE))[season]['uq_bounds']))
    return [n for n,_ in nb],[list(map(float,b)) for _,b in nb]
def evaluate(X):
    c=lambda j: X[:,j:j+1]
    prm=dict(m=c(0)+c(1),Cd=c(2),A=c(3),Cr=c(4),eta_peak=c(5),eta_regen=c(6),P_aux=c(7),R_int=c(8),
             P_regen_max=c(9)*1e3,P_motor_max=c(10)*1e3,E_use=c(14)*c(11)*3.6e6,rho=c(13),
             rw=np.full_like(c(0),lt['wheel_radius_m']['value']),gr=np.full_like(c(0),lt['gear_ratio']['value']))
    g=c(12); gradeT=np.where(dT[None,:]<=dT[-1]/2,g,-g)
    E1,d1=simulate_batch(vT,np.zeros_like(c(0)),prm); E2,_=simulate_batch(vT,gradeT,prm); E3,d3=simulate_batch(vW,np.zeros_like(c(0)),prm)
    cTf=E1/(d1/1000)*100; cTt=E2/(d1/1000)*100; cWf=E3/(d3/1000)*100
    if X.shape[1]>15:                          # bias-corrected modes: correct kinematics, keep grade effect
        corr=cTf*X[:,15]; cTf=cTf+corr; cTt=cTt+corr
    Euse=X[:,14]*X[:,11]
    rWf=Euse/(cWf/100); rTt=Euse/(cTt/100)
    return np.column_stack([cTf,cTt,cWf,cTf-cWf,cTt-cTf,rWf-rTt,rWf,rTt,(rWf-rTt)/rTt*100])
OUTS=['cons_TETDC_flat','cons_TETDC_tehran','cons_WLTC_flat','shape_effect','topo_effect','range_overestimate_km','range_WLTC_flat_km','range_TETDC_tehran_km','range_overestimate_pct']
def run(mode,season,part=0):
    CH=2000
    bc=mode.endswith('_bc')
    names,bounds=names_bounds(season,bc)
    f=os.path.join(OUTDIR,f'{mode}_{season}_X.npy')
    if not os.path.exists(f):
        if mode.startswith('mc'):
            r=np.random.default_rng(20261003); lo=np.array([b[0] for b in bounds]); hi=np.array([b[1] for b in bounds])
            X=lo+(hi-lo)*r.random((10000,len(bounds)))
        else:
            X=sobol_sample.sample({'num_vars':len(names),'names':names,'bounds':bounds},1024,calc_second_order=False,seed=20261003)
        np.save(f,X)
    X=np.load(f); n=int(np.ceil(len(X)/CH))
    for p in range(part, n):
        of=os.path.join(OUTDIR,f'{mode}_{season}_Y{p:03d}.npy')
        if os.path.exists(of): continue
        np.save(of,evaluate(X[p*CH:(p+1)*CH])); print(mode,season,'chunk',p+1,'/',n,flush=True)

if __name__=='__main__':
    modes=[sys.argv[1]] if len(sys.argv)>1 else ['mc','sobol','mc_bc','sobol_bc']
    seasons=[sys.argv[2]] if len(sys.argv)>2 else ['summer','winter','shoulder']
    for m in modes:
        for s in seasons: run(m,s)
