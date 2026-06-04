"""
Hybrid Imputer: Linear Interpolation for interior points, Ridge for edge points
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df_original = pd.read_csv('dataset.csv')
df = df_original.dropna(subset=['datetime']).reset_index(drop=True)

df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df['time_of_day'] = df['datetime_dt'].dt.hour * 60 + df['datetime_dt'].dt.minute
expiry = pd.to_datetime('2026-01-27 15:30')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600)
df['dte'] = df['dte'].clip(lower=0.001)

option_cols = [c for c in df_original.columns if c.startswith('NIFTY')]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])

strikes_ce = np.array([float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols])
strikes_pe = np.array([float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols])
base_aux = ['underlying_price', 'time_of_day', 'dte']

def validate(n_trials=3):
    scores = []
    scores_ridge = []
    scores_linear = []
    
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
        
        # 1. Get Ridge predictions
        all_cols = base_aux + option_cols
        imp = IterativeImputer(estimator=Ridge(alpha=0.05), max_iter=30, tol=1e-5, random_state=42)
        filled_ridge_matrix = imp.fit_transform(masked[all_cols])
        filled_ridge = pd.DataFrame(filled_ridge_matrix, columns=all_cols)[option_cols].clip(lower=0.001)
        
        # 2. Get Linear Extrapolation predictions
        filled_linear = masked[option_cols].copy()
        
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
                min_k, max_k = min(ce_valid_strikes), max(ce_valid_strikes)
                for j, c in enumerate(ce_cols):
                    if pd.isna(row[c]):
                        k = strikes_ce[j]
                        if k >= min_k and k <= max_k:
                            # INTERPOLATION: use linear
                            pred = float(f_ce(k))
                            filled_linear.at[ix, c] = np.clip(pred, 0.01, 6.0)
                        else:
                            # EXTRAPOLATION: use Ridge
                            filled_linear.at[ix, c] = filled_ridge.at[ix, c]
            else:
                for c in ce_cols:
                    if pd.isna(row[c]):
                        filled_linear.at[ix, c] = filled_ridge.at[ix, c]

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
                min_k, max_k = min(pe_valid_strikes), max(pe_valid_strikes)
                for j, c in enumerate(pe_cols):
                    if pd.isna(row[c]):
                        k = strikes_pe[j]
                        if k >= min_k and k <= max_k:
                            # INTERPOLATION: use linear
                            pred = float(f_pe(k))
                            filled_linear.at[ix, c] = np.clip(pred, 0.01, 6.0)
                        else:
                            # EXTRAPOLATION: use Ridge
                            filled_linear.at[ix, c] = filled_ridge.at[ix, c]
            else:
                for c in pe_cols:
                    if pd.isna(row[c]):
                        filled_linear.at[ix, c] = filled_ridge.at[ix, c]
                        
        
        y_true, y_pred = [], []
        y_pred_ridge = []
        for ix, c, tv in truth:
            pv = filled_linear.at[ix, c]
            if pd.notna(pv) and pd.notna(tv):
                y_true.append(tv)
                y_pred.append(pv)
                y_pred_ridge.append(filled_ridge.at[ix, c])
                
        mse = mean_squared_error(y_true, y_pred)
        mse_ridge = mean_squared_error(y_true, y_pred_ridge)
        scores.append(mse)
        scores_ridge.append(mse_ridge)
    
    print(f"Hybrid Imputer Mean CV: {np.mean(scores):.10f}")
    print(f"Pure Ridge Mean CV    : {np.mean(scores_ridge):.10f}")

print("Testing Hybrid Imputer (Linear Interpolation + Ridge Extrapolation)...")
validate()
