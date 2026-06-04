"""
Evaluate Deterministic Cross-Sectional Linear Extrapolation
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

def linear_extrapolate(masked):
    filled = masked[option_cols].copy()
    
    for ix, row in masked.iterrows():
        # CE
        ce_valid_strikes = []
        ce_valid_ivs = []
        for j, c in enumerate(ce_cols):
            val = row[c]
            if pd.notna(val):
                ce_valid_strikes.append(strikes_ce[j])
                ce_valid_ivs.append(val)
                
        if len(ce_valid_strikes) >= 2:
            f_ce = interp1d(ce_valid_strikes, ce_valid_ivs, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]):
                    pred = float(f_ce(strikes_ce[j]))
                    filled.at[ix, c] = np.clip(pred, 0.01, 6.0)
        elif len(ce_valid_strikes) == 1:
            for c in ce_cols:
                if pd.isna(row[c]):
                    filled.at[ix, c] = ce_valid_ivs[0]

        # PE
        pe_valid_strikes = []
        pe_valid_ivs = []
        for j, c in enumerate(pe_cols):
            val = row[c]
            if pd.notna(val):
                pe_valid_strikes.append(strikes_pe[j])
                pe_valid_ivs.append(val)
                
        if len(pe_valid_strikes) >= 2:
            f_pe = interp1d(pe_valid_strikes, pe_valid_ivs, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]):
                    pred = float(f_pe(strikes_pe[j]))
                    filled.at[ix, c] = np.clip(pred, 0.01, 6.0)
        elif len(pe_valid_strikes) == 1:
            for c in pe_cols:
                if pd.isna(row[c]):
                    filled.at[ix, c] = pe_valid_ivs[0]
                    
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
        
        filled = linear_extrapolate(masked)
        
        y_true, y_pred = [], []
        for ix, c, tv in truth:
            pv = filled.at[ix, c]
            if pd.notna(pv) and pd.notna(tv):
                y_true.append(tv)
                y_pred.append(pv)
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
    
    mu = np.mean(scores)
    print(f"Cross-Sectional Linear Extrapolation : {mu:.10f}")
    return mu

print("Evaluating Cross-Sectional Linear Extrapolation...")
validate()
