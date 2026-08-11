"""Model 3D — SHAP analysis of the MLP.

For a neural network, model-agnostic SHAP is used here. To keep runtime
reasonable on the small competition dataset, the explanation set is sampled.
The script reports mean absolute SHAP by transformed feature and by original
feature family (Age, Partner, Industry, State, Country).
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import shap
from common import FEATURES, load_raw, make_preprocessor, target_table


def run(data_csv, target='placement', seed=42, background_n=80, explain_n=120, out_csv=None):
    df=load_raw(data_csv); t=target_table(df,target)
    X=t[FEATURES]; y=t['y']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=seed)
    model=Pipeline([('prep',make_preprocessor(t)),('mlp',MLPRegressor(hidden_layer_sizes=(64,32),alpha=1e-3,max_iter=2000,early_stopping=True,random_state=seed))])
    model.fit(Xtr,ytr)
    Xt=model.named_steps['prep'].transform(Xtr)
    Xe=model.named_steps['prep'].transform(Xte)
    rng=np.random.default_rng(seed)
    bg=Xt[rng.choice(len(Xt),size=min(background_n,len(Xt)),replace=False)]
    ex=Xe[rng.choice(len(Xe),size=min(explain_n,len(Xe)),replace=False)]
    predict=lambda z:model.named_steps['mlp'].predict(z)
    explainer=shap.Explainer(predict,bg,algorithm='permutation')
    sv=explainer(ex,max_evals=2*ex.shape[1]+1)
    feature_names=model.named_steps['prep'].get_feature_names_out()
    imp=pd.DataFrame({'transformed_feature':feature_names,'mean_abs_shap':np.abs(sv.values).mean(axis=0)})
    imp['family']=imp['transformed_feature'].str.replace(r'^num__.*','Age',regex=True)
    for original in ['ballroom_partner','celebrity_industry','celebrity_homestate','celebrity_homecountry/region']:
        imp.loc[imp.transformed_feature.str.contains(original,regex=False),'family']=original
    fam=imp.groupby('family',as_index=False)['mean_abs_shap'].sum().sort_values('mean_abs_shap',ascending=False)
    print(fam.to_string(index=False))
    if out_csv: fam.to_csv(out_csv,index=False)
    return model,imp,fam

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--target',default='placement',choices=['placement','avg_judge_score','fan_support']); ap.add_argument('--out-csv')
    a=ap.parse_args(); run(a.data,a.target,out_csv=a.out_csv)
