"""Run MLR, Random Forest, and MLP side-by-side for each available Model 3 target."""
import argparse
from pathlib import Path
import pandas as pd
from importlib import import_module

TARGETS=['placement','avg_judge_score','fan_support']

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out-dir',default='outputs/model3_compare'); a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    mlr=import_module('01_mlr')
    rf=import_module('02_random_forest')
    mlp=import_module('03_mlp')
    for target in TARGETS:
        try:
            _,r1=mlr.run(a.data,target,out_csv=out/f'mlr_{target}.csv')
            _,r2,_=rf.run(a.data,target,out_csv=out/f'rf_{target}.csv',importance_csv=out/f'rf_mdi_{target}.csv')
            _,r3=mlp.run(a.data,target,out_csv=out/f'mlp_{target}.csv')
            for method,r in [('MLR',r1),('RandomForest',r2),('MLP',r3)]: rows.append({'target':target,'model':method,**r})
        except ValueError as e:
            print(f'Skipping {target}: {e}')
    pd.DataFrame(rows).to_csv(out/'model_comparison.csv',index=False)
