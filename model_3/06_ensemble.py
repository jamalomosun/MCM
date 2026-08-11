"""Model 3E — RF + MLP predictive ensemble.

The paper compares RF and MLP; it does not combine them. This extension fits
both models and learns a convex weight w so that

    y_hat_ensemble = w*y_hat_RF + (1-w)*y_hat_MLP.

Weights are chosen using only the training portion via K-fold cross-validation,
then evaluated once on a held-out test set.
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from common import FEATURES, load_raw, make_preprocessor, target_table


def make_models(t, seed):
    return (
        Pipeline([('prep',make_preprocessor(t)),('rf',RandomForestRegressor(n_estimators=500,random_state=seed,n_jobs=-1))]),
        Pipeline([('prep',make_preprocessor(t)),('mlp',MLPRegressor(hidden_layer_sizes=(64,32),alpha=1e-3,max_iter=2000,early_stopping=True,random_state=seed))])
    )

def run(data_csv,target='placement',seed=42,test_size=0.2,n_splits=5,out_csv=None):
    df=load_raw(data_csv); t=target_table(df,target); X=t[FEATURES]; y=t['y']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=seed)
    rf,mlp=make_models(t,seed)
    kf=KFold(n_splits=n_splits,shuffle=True,random_state=seed)
    rf_cv=cross_val_predict(rf,Xtr,ytr,cv=kf,n_jobs=None); mlp_cv=cross_val_predict(mlp,Xtr,ytr,cv=kf,n_jobs=None)
    def cv_loss(w): return mean_squared_error(ytr,w*rf_cv+(1-w)*mlp_cv)
    opt=minimize_scalar(cv_loss,bounds=(0,1),method='bounded')
    w=float(opt.x)
    rf.fit(Xtr,ytr); mlp.fit(Xtr,ytr)
    prf=rf.predict(Xte); pmlp=mlp.predict(Xte); penc=w*prf+(1-w)*pmlp
    metrics=[]
    for name,p in [('RF',prf),('MLP',pmlp),('Ensemble',penc)]:
        metrics.append({'target':target,'model':name,'r2':r2_score(yte,p),'mae':mean_absolute_error(yte,p),'rmse':mean_squared_error(yte,p)**0.5})
    result={'target':target,'rf_weight':w,'mlp_weight':1-w,'cv_mse':float(opt.fun)}
    print(json.dumps(result,indent=2)); print(pd.DataFrame(metrics).to_string(index=False))
    if out_csv: pd.DataFrame({'y_true':yte,'rf_pred':prf,'mlp_pred':pmlp,'ensemble_pred':penc}).to_csv(out_csv,index=False)
    return rf,mlp,result,pd.DataFrame(metrics)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--target',default='placement',choices=['placement','avg_judge_score','fan_support']); ap.add_argument('--out-csv'); a=ap.parse_args(); run(a.data,a.target,out_csv=a.out_csv)
