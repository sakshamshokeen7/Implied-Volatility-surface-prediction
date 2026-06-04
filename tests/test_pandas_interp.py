"""
Test simple pandas interpolation across strikes (axis=1).
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df_original = pd.read_csv('dataset.csv')
df = df_original.dropna(subset=['datetime']).reset_index(drop=True)

option_cols = [c for c in df.columns if c.startswith('NIFTY')]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])

def pandas_interp(masked, method='spline', order=2):
    filled = masked.copy()
    
    # CE
    ce_df = filled[ce_cols].copy()
    ce_df = ce_df.interpolate(method=method, order=order, axis=1, limit_direction='both')
    
    # PE
    pe_df = filled[pe_cols].copy()
    pe_df = pe_df.interpolate(method=method, order=order, axis=1, limit_direction='both')
    
    filled[ce_cols] = ce_df
    filled[pe_cols] = pe_df
    
    # Fill remaining NaNs across time just in case
    filled[option_cols] = filled[option_cols].interpolate(method='linear', axis=0, limit_direction='both')
    filled[option_cols] = filled[option_cols].ffill()
    
    return filled[option_cols].clip(lower=0.001)

def validate(method, order=None, n_trials=3):
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
        
        filled = pandas_interp(masked, method=method, order=order)
        
        y_true, y_pred = [], []
        for ix, c, tv in truth:
            pv = filled.at[ix, c]
            if pd.notna(pv) and pd.notna(tv):
                y_true.append(tv)
                y_pred.append(pv)
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
    
    mu = np.mean(scores)
    print(f"Pandas Interp ({method}, order={order}): {mu:.10f}")
    return mu

print("Testing simple pandas axis=1 interpolation...")
validate('linear')
validate('spline', order=2)
validate('spline', order=3)
validate('polynomial', order=2)
validate('polynomial', order=3)
validate('akima')
validate('pchip')
