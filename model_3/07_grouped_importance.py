"""Model 3F — grouped feature importance and permutation importance.

This is a methodological improvement over raw MDI: one-hot columns belonging
to the same original feature family are recombined, and permutation importance
is calculated on held-out data. This helps reduce the tendency of raw split-based
importance to over-fragment categorical variables.
"""
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from common import FEATURES, load_raw, make_preprocessor, target_table


def family(name):
    if name.startswith('num__'): return 'Age'
    for x,label in [
        ('ballroom_partner','Partner'),('celebrity_industry','Industry'),
        ('celebrity_homestate','State'),('celebrity_homecountry/region','Country')]:
        if x in name: return label
    return name

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--target',default='placement',choices=['placement','avg_judge_score']); ap.add_argument('--out-csv',default='grouped_importance.csv'); a=ap.parse_args()
    df=load_raw(a.data); t=target_table(df,a.target)
    X=t[FEATURES]; y=t.y
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    m=Pipeline([('prep',make_preprocessor(t)),('rf',RandomForestRegressor(n_estimators=500,random_state=42,n_jobs=-1))])
    m.fit(Xtr,ytr)
    names=m.named_steps['prep'].get_feature_names_out()
    mdi=pd.DataFrame({'family':[family(n) for n in names],'mdi':m.named_steps['rf'].feature_importances_}).groupby('family',as_index=False).sum()
    # Permutation importance is computed on the original feature families by permuting raw columns through the pipeline.
    perm=permutation_importance(m,Xte,yte,n_repeats=30,random_state=42,scoring='neg_mean_squared_error')
    p=pd.DataFrame({'family':FEATURES,'permutation_mse_increase':perm.importances_mean})
    mapping=dict(zip(FEATURES,['Age','Partner','Industry','State','Country']))
    p['family']=p['family'].map(mapping)
    out=mdi.merge(p,on='family').sort_values('permutation_mse_increase',ascending=False)
    out.to_csv(a.out_csv,index=False); print(out.to_string(index=False))
