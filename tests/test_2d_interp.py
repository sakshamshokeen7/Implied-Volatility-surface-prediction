"""
Test Time Interpolation + Cross-Sectional Interpolation
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
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

def validate_model(time_interp=False, cross_method='linear', fallback='extrapolate'):
    scores = []
    n_trials = 3
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
                
        filled = masked.copy()
        
        # 1. Time Interpolation
        if time_interp:
            for c in option_cols:
                filled[c] = filled[c].interpolate(method='linear', limit_direction='both')
                
        # 2. Cross-Sectional Interpolation
        for ix, row in filled.iterrows():
            # CE
            ce_valid_strikes = []
            ce_valid_ivs = []
            for j, c in enumerate(ce_cols):
                val = row[c]
                if pd.notna(val):
                    ce_valid_strikes.append(strikes_ce[j])
                    ce_valid_ivs.append(val)
                    
            if len(ce_valid_strikes) >= 2:
                if cross_method == 'linear':
                    f_ce = interp1d(ce_valid_strikes, ce_valid_ivs, kind='linear', fill_value=fallback)
                elif cross_method == 'pchip':
                    f_ce = PchipInterpolator(ce_valid_strikes, ce_valid_ivs, extrapolate=(fallback=='extrapolate'))
                
                for j, c in enumerate(ce_cols):
                    if pd.isna(row[c]):
                        try:
                            pred = float(f_ce(strikes_ce[j]))
                            filled.at[ix, c] = np.clip(pred, 0.01, 6.0)
                        except:
                            pass
                            
            # PE
            pe_valid_strikes = []
            pe_valid_ivs = []
            for j, c in enumerate(pe_cols):
                val = row[c]
                if pd.notna(val):
                    pe_valid_strikes.append(strikes_pe[j])
                    pe_valid_ivs.append(val)
                    
            if len(pe_valid_strikes) >= 2:
                if cross_method == 'linear':
                    f_pe = interp1d(pe_valid_strikes, pe_valid_ivs, kind='linear', fill_value=fallback)
                elif cross_method == 'pchip':
                    f_pe = PchipInterpolator(pe_valid_strikes, pe_valid_ivs, extrapolate=(fallback=='extrapolate'))
                
                for j, c in enumerate(pe_cols):
                    if pd.isna(row[c]):
                        try:
                            pred = float(f_pe(strikes_pe[j]))
                            filled.at[ix, c] = np.clip(pred, 0.01, 6.0)
                        except:
                            pass
                            
        # Forward/Backward fill any remaining
        filled = filled.ffill()
        
        y_true, y_pred = [], []
        for ix, c, tv in truth:
            pv = filled.at[ix, c]
            if pd.notna(pv) and pd.notna(tv):
                y_true.append(tv)
                y_pred.append(pv)
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
        
    print(f"Time: {str(time_interp):5s} | Cross: {cross_method:6s} | CV MSE: {np.mean(scores):.10f}")

print("Evaluating 2D Interpolation Strategies...")
validate_model(time_interp=False, cross_method='linear', fallback='extrapolate')
validate_model(time_interp=False, cross_method='pchip', fallback='extrapolate')
validate_model(time_interp=True, cross_method='linear', fallback='extrapolate')
validate_model(time_interp=True, cross_method='pchip', fallback='extrapolate')
