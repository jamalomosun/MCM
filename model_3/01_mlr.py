"""Model 3A — Multiple Linear Regression.

This is the linear baseline described in the paper. It is intentionally kept
separate from the nonlinear models so the failure of a linear specification can
be measured rather than assumed.
"""
from pathlib import Path
import argparse
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from common import FEATURES, load_raw, make_preprocessor, target_table


def run(data_csv, target='placement', test_size=0.2, seed=42, out_csv=None):
    df=load_raw(data_csv)
    t=target_table(df,target)
    X=t[FEATURES]; y=t['y']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=seed)
    model=Pipeline([('prep',make_preprocessor(t)),('model',LinearRegression())])
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    result={'target':target,'r2':r2_score(yte,pred),'mae':mean_absolute_error(yte,pred),
            'rmse':mean_squared_error(yte,pred)**0.5,'n_train':len(Xtr),'n_test':len(Xte)}
    if out_csv:
        pd.DataFrame({'y_true':yte,'y_pred':pred}).to_csv(out_csv,index=False)
    print(json.dumps(result,indent=2))
    return model,result

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--target',default='placement',choices=['placement','avg_judge_score','fan_support']); ap.add_argument('--out-csv')
    a=ap.parse_args(); run(a.data,a.target,out_csv=a.out_csv)
