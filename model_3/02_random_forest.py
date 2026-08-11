"""Model 3B — Random Forest regression and MDI feature importance."""
from pathlib import Path
import argparse, json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from common import FEATURES, load_raw, make_preprocessor, target_table


def run(data_csv, target='placement', test_size=0.2, seed=42, n_estimators=500, out_csv=None, importance_csv=None):
    df=load_raw(data_csv); t=target_table(df,target)
    X=t[FEATURES]; y=t['y']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=seed)
    prep=make_preprocessor(t)
    model=Pipeline([('prep',prep),('rf',RandomForestRegressor(n_estimators=n_estimators,random_state=seed,n_jobs=-1))])
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    names=model.named_steps['prep'].get_feature_names_out()
    imps=pd.DataFrame({'transformed_feature':names,'mdi':model.named_steps['rf'].feature_importances_}).sort_values('mdi',ascending=False)
    result={'target':target,'r2':r2_score(yte,pred),'mae':mean_absolute_error(yte,pred),'rmse':mean_squared_error(yte,pred)**0.5,'n_train':len(Xtr),'n_test':len(Xte)}
    if out_csv: pd.DataFrame({'y_true':yte,'y_pred':pred}).to_csv(out_csv,index=False)
    if importance_csv: imps.to_csv(importance_csv,index=False)
    print(json.dumps(result,indent=2)); print(imps.head(15).to_string(index=False))
    return model,result,imps

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--target',default='placement',choices=['placement','avg_judge_score','fan_support']); ap.add_argument('--out-csv'); ap.add_argument('--importance-csv')
    a=ap.parse_args(); run(a.data,a.target,out_csv=a.out_csv,importance_csv=a.importance_csv)
