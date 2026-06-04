"""
Joint CE/PE interpolation.
Options at the same strike and expiry must have the same implied volatility.
By pooling CE and PE strikes together, we get a much wider and denser smile!
"""
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
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

def joint_smile_impute(masked):
    filled = masked[option_cols].copy()
    
    for ix, row in masked.iterrows():
        S = row['underlying_price']
        
        # Pool all valid IVs for this row
        valid_strikes = []
        valid_ivs = []
        
        for i, c in enumerate(ce_cols):
            val = row[c]
            if pd.notna(val):
                valid_strikes.append(strikes_ce[i])
                valid_ivs.append(val)
                
        for i, c in enumerate(pe_cols):
            val = row[c]
            if pd.notna(val):
                valid_strikes.append(strikes_pe[i])
                valid_ivs.append(val)
                
        valid_strikes = np.array(valid_strikes)
        valid_ivs = np.array(valid_ivs)
        
        if len(valid_strikes) > 3:
            # Sort by strike
            sort_idx = np.argsort(valid_strikes)
            valid_strikes = valid_strikes[sort_idx]
            valid_ivs = valid_ivs[sort_idx]
            
            # Remove duplicates (if any CE and PE overlap)
            unique_strikes, unique_idx = np.unique(valid_strikes, return_index=True)
            unique_ivs = valid_ivs[unique_idx]
            
            # Fit PCHIP
            interp = PchipInterpolator(unique_strikes, unique_ivs)
            
            # Fill missing CE
            for i, c in enumerate(ce_cols):
                if pd.isna(row[c]):
                    k = strikes_ce[i]
                    if k <= unique_strikes[0]: val = unique_ivs[0]
                    elif k >= unique_strikes[-1]: val = unique_ivs[-1]
                    else: val = float(interp(k))
                    filled.at[ix, c] = max(0.001, val)
                    
            # Fill missing PE
            for i, c in enumerate(pe_cols):
                if pd.isna(row[c]):
                    k = strikes_pe[i]
                    if k <= unique_strikes[0]: val = unique_ivs[0]
                    elif k >= unique_strikes[-1]: val = unique_ivs[-1]
                    else: val = float(interp(k))
                    filled.at[ix, c] = max(0.001, val)
                    
    # Fill remaining NaNs with column mean
    for c in option_cols:
        if filled[c].isna().any():
            filled[c] = filled[c].fillna(filled[c].mean())
            
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

print("Testing Joint CE/PE PCHIP Interpolation...")
validate(joint_smile_impute, "Joint PCHIP Interpolation")
