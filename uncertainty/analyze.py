"""Summarise Monte-Carlo statistics and Sobol' indices -> data/processed/uq_results.json (original)
and data/processed/uq_results_bc.json (with the reconstruction-bias correction)."""
import sys,os,json,numpy as np,glob
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); sys.path.insert(0,HERE)
D=os.path.join(ROOT,'data','processed','uncertainty')
from uq import names_bounds, OUTS
from SALib.analyze import sobol
def summarise(tag):
  res={}
  for s in ['summer','winter','shoulder']:
    Y=np.vstack([np.load(f) for f in sorted(glob.glob(os.path.join(D,f'mc{tag}_{s}_Y*.npy')))])
    r={}
    for j,o in enumerate(OUTS):
        y=Y[:,j]; r[o]=dict(mean=round(float(y.mean()),2),median=round(float(np.median(y)),2),
            p2_5=round(float(np.percentile(y,2.5)),2),p97_5=round(float(np.percentile(y,97.5)),2),
            prob_positive=round(float((y>0).mean()),4))
    names,bounds=names_bounds(s,bc=(tag=='_bc'))
    Ys=np.vstack([np.load(f) for f in sorted(glob.glob(os.path.join(D,f'sobol{tag}_{s}_Y*.npy')))])
    prob={'num_vars':len(names),'names':names,'bounds':bounds}
    sob={}
    for o in ['range_overestimate_km','range_overestimate_pct','cons_TETDC_tehran','topo_effect','shape_effect']:
        Si=sobol.analyze(prob,Ys[:,OUTS.index(o)],calc_second_order=False,seed=20261003)
        sob[o]={n:dict(S1=round(float(a),3),S1_ci=round(float(b),3),ST=round(float(c),3),ST_ci=round(float(d),3)) for n,a,b,c,d in zip(names,Si['S1'],Si['S1_conf'],Si['ST'],Si['ST_conf'])}
    res[s]=dict(n_mc=int(len(Y)),n_sobol=int(len(Ys)),monte_carlo=r,sobol=sob)
  json.dump(res,open(os.path.join(ROOT,'data','processed',f'uq_results{tag}.json'),'w'),indent=1)
  return res
for tag in ['','_bc']:
  res=summarise(tag); print(f"\n##### uq_results{tag}.json")
  for s in res:
      m=res[s]['monte_carlo']; print(f"\n=== {s.upper()} (MC n={res[s]['n_mc']}, Sobol n={res[s]['n_sobol']})")
      for o in OUTS: print(f"  {o:24s} mean {m[o]['mean']:8.2f}  95% [{m[o]['p2_5']:7.2f}, {m[o]['p97_5']:7.2f}]  P(>0)={m[o]['prob_positive']:.3f}")
      for o in ['range_overestimate_km','topo_effect']:
          top=sorted(res[s]['sobol'][o].items(),key=lambda kv:-kv[1]['ST'])[:5]
          print(f"  Sobol ST [{o}]: "+", ".join(f"{n} {v['ST']:.2f}±{v['ST_ci']:.2f}" for n,v in top))
