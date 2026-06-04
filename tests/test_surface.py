"""
2D Surface Interpolation (Time x Strike) for IV Surface.
"""
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df_original = pd.read_csv('dataset.csv')
df = df_original.dropna(subset=['datetime']).reset_index(drop=True)

df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df['time'] = (df['datetime_dt'] - df['datetime_dt'].min()).dt.total_seconds() / 3600.0

option_cols = [c for c in df.columns if c.startswith('NIFTY')]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])

strikes_ce = np.array([float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols])
strikes_pe = np.array([float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols])

def surface_impute(masked):
    filled = masked[option_cols].copy()
    
    # CE Surface
    pts_ce = []
    vals_ce = []
    for ix, row in masked.iterrows():
        t = row['time']
        for j, c in enumerate(ce_cols):
            v = row[c]
            if pd.notna(v):
                pts_ce.append([t, strikes_ce[j]])
                vals_ce.append(v)
    pts_ce = np.array(pts_ce)
    vals_ce = np.array(vals_ce)
    
    # PE Surface
    pts_pe = []
    vals_pe = []
    for ix, row in masked.iterrows():
        t = row['time']
        for j, c in enumerate(pe_cols):
            v = row[c]
            if pd.notna(v):
                pts_pe.append([t, strikes_pe[j]])
                vals_pe.append(v)
    pts_pe = np.array(pts_pe)
    vals_pe = np.array(vals_pe)
    
    # Interpolate CE
    missing_ce_pts = []
    missing_ce_idx = []
    for ix, row in masked.iterrows():
        t = row['time']
        for j, c in enumerate(ce_cols):
            if pd.isna(row[c]):
                missing_ce_pts.append([t, strikes_ce[j]])
                missing_ce_idx.append((ix, c))
    
    if len(missing_ce_pts) > 0:
        missing_ce_pts = np.array(missing_ce_pts)
        preds_ce = griddata(pts_ce, vals_ce, missing_ce_pts, method='linear')
        
        # fallback to nearest if outside hull
        nans = np.isnan(preds_ce)
        if nans.any():
            preds_ce[nans] = griddata(pts_ce, vals_ce, missing_ce_pts[nans], method='nearest')
            
        for k, (ix, c) in enumerate(missing_ce_idx):
            filled.at[ix, c] = max(0.001, preds_ce[k])

    # Interpolate PE
    missing_pe_pts = []
    missing_pe_idx = []
    for ix, row in masked.iterrows():
        t = row['time']
        for j, c in enumerate(pe_cols):
            if pd.isna(row[c]):
                missing_pe_pts.append([t, strikes_pe[j]])
                missing_pe_idx.append((ix, c))
                
    if len(missing_pe_pts) > 0:
        missing_pe_pts = np.array(missing_pe_pts)
        preds_pe = griddata(pts_pe, vals_pe, missing_pe_pts, method='linear')
        
        nans = np.isnan(preds_pe)
        if nans.any():
            preds_pe[nans] = griddata(pts_pe, vals_pe, missing_pe_pts[nans], method='nearest')
            
        for k, (ix, c) in enumerate(missing_pe_idx):
            filled.at[ix, c] = max(0.001, preds_pe[k])
            
    return filled

def validate(impute_fn, name, n_trials=3):
    scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 101 + 13)
        masked = df.copy()
        truth = []
        for c in option_cols:
            obs = df.index[df[c].notna()].tolist()
            k = max(1, int(len(obs) * 0.15))
            hide = rng.choice(obs, size=k, replace=False)
            for ix in hide:
                truth.append((ix, c, df.at[ix, c]))
                masked.at[ix, c] = np.nan
        
        filled = impute_fn(masked)
        
        y_true, y_pred = [], []
        for ix, c, tv in truth:
            pv = filled.at[ix, c]
            if pd.notna(pv) and pd.notna(tv):
                y_true.append(tv)
                y_pred.append(pv)
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
    mu = np.mean(scores)
    print(f"  {name:40s} : {mu:.10f}")
    return mu

print("Testing 2D Surface Interpolation...")
validate(surface_impute, "2D Linear/Nearest Surface")
