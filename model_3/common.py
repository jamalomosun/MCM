"""Shared preprocessing for Model 3.

Our paper's Model 3 uses five contestant attributes:
- age
- ballroom partner
- industry
- home state
- home country/region

It evaluates three response variables when fan estimates are available:
- final placement
- average judge score
- fan support / fan percentage estimate

"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    'celebrity_age_during_season',
    'ballroom_partner',
    'celebrity_industry',
    'celebrity_homestate',
    'celebrity_homecountry/region',
]


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'placement' not in df.columns:
        raise ValueError("Expected 'placement' column in the competition CSV.")
    return df


def judge_long_form(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _, r in df.iterrows():
        for wk in range(1, 12):
            cols=[f'week{wk}_judge{k}_score' for k in range(1,5)]
            vals=[r[c] for c in cols if c in df.columns and pd.notna(r[c])]
            if not vals:
                continue
            rows.append({
                'season': int(r['season']),
                'week': wk,
                'celebrity_name': r['celebrity_name'],
                'judge_average': float(np.mean(vals)),
                'judge_total': float(np.sum(vals)),
            })
    return pd.DataFrame(rows)


def contestant_level(df: pd.DataFrame, fan_csv: str | Path | None = None) -> pd.DataFrame:
    out=df.copy()
    jl=judge_long_form(df)
    judge_avg=(jl.groupby(['season','celebrity_name'], as_index=False)
                 .agg(avg_judge_score=('judge_average','mean')))
    out=out.merge(judge_avg, on=['season','celebrity_name'], how='left')
    # Placeholders for Model 2 handoff.
    out['fan_support'] = np.nan
    if fan_csv is not None:
        fan= pd.read_csv(fan_csv)
        required={'season','celebrity_name','fan_support'}
        missing=required-set(fan.columns)
        if missing:
            raise ValueError(f'Fan file missing columns: {sorted(missing)}')
        # Accept either weekly rows or contestant-level summaries.
        if {'week'}.issubset(fan.columns):
            fan=(fan.groupby(['season','celebrity_name'], as_index=False)
                    .agg(fan_support=('fan_support','mean')))
        out=out.drop(columns=['fan_support']).merge(fan, on=['season','celebrity_name'], how='left')
    return out


def make_preprocessor(df: pd.DataFrame):
    cats=[c for c in FEATURES if c != 'celebrity_age_during_season']
    nums=['celebrity_age_during_season']
    try:
        ohe=OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe=OneHotEncoder(handle_unknown='ignore', sparse=False)
    return ColumnTransformer([
        ('num', Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]), nums),
        ('cat', Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',ohe)]), cats),
    ], remainder='drop')


def target_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    t=contestant_level(df)
    if target not in {'placement','avg_judge_score','fan_support'}:
        raise ValueError(target)
    if target=='placement':
        t['y']=pd.to_numeric(t['placement'], errors='coerce')
    else:
        t['y']=pd.to_numeric(t[target], errors='coerce')
    return t.dropna(subset=['y']).copy()
