"""
Evaluate Joint CE/PE Deterministic Cross-Sectional Linear Extrapolation
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df_original = pd.read_csv('dataset.csv')
df = df_original.dropna(subset=['datetime']).reset_index(drop=True)

option_cols = [c for c in df_original.columns if c.startswith('NIFTY')]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])

strikes_ce = np.array([float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols])
strikes_pe = np.array([float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols])

def joint_linear_extrapolate(masked):
    filled = masked[option_cols].copy()
    
    for ix, row in masked.iterrows():
        valid_strikes = []
        valid_ivs = []
        
        # Collect all valid IVs from CE and PE
        for j, c in enumerate(ce_cols):
            val = row[c]
            if pd.notna(val):
                valid_strikes.append(strikes_ce[j])
                valid_ivs.append(val)
                
        for j, c in enumerate(pe_cols):
            val = row[c]
            if pd.notna(val):
                valid_strikes.append(strikes_pe[j])
                valid_ivs.append(val)
                
        valid_strikes = np.array(valid_strikes)
        valid_ivs = np.array(valid_ivs)
        
        if len(valid_strikes) >= 2:
            # Sort them
            sort_idx = np.argsort(valid_strikes)
            valid_strikes = valid_strikes[sort_idx]
            valid_ivs = valid_ivs[sort_idx]
            
            # Deduplicate (average if same strike exists in both CE and PE)
            unique_strikes = np.unique(valid_strikes)
            unique_ivs = []
            for k in unique_strikes:
                unique_ivs.append(np.mean(valid_ivs[valid_strikes == k]))
            unique_ivs = np.array(unique_ivs)
            
            if len(unique_strikes) >= 2:
                f = interp1d(unique_strikes, unique_ivs, kind='linear', fill_value='extrapolate')
                
                # Predict missing CE
                for j, c in enumerate(ce_cols):
                    if pd.isna(row[c]):
                        pred = float(f(strikes_ce[j]))
                        filled.at[ix, c] = np.clip(pred, 0.01, 6.0)
                        
                # Predict missing PE
                for j, c in enumerate(pe_cols):
                    if pd.isna(row[c]):
                        pred = float(f(strikes_pe[j]))
                        filled.at[ix, c] = np.clip(pred, 0.01, 6.0)
            else:
                for c in option_cols:
                    if pd.isna(row[c]):
                        filled.at[ix, c] = unique_ivs[0]
                        
        elif len(valid_strikes) == 1:
            for c in option_cols:
                if pd.isna(row[c]):
                    filled.at[ix, c] = valid_ivs[0]
                    
    # Global fallback for any remaining NaNs
    for c in option_cols:
        if filled[c].isna().any():
            filled[c] = filled[c].fillna(filled[c].mean())
            
    return filled

def validate(n_trials=3):
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
        
        filled = joint_linear_extrapolate(masked)
        
        y_true, y_pred = [], []
        for ix, c, tv in truth:
            pv = filled.at[ix, c]
            if pd.notna(pv) and pd.notna(tv):
                y_true.append(tv)
                y_pred.append(pv)
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
    
    mu = np.mean(scores)
    print(f"Joint Linear Extrapolation : {mu:.10f}")
    return mu

print("Evaluating Joint CE/PE Linear Extrapolation...")
validate()
