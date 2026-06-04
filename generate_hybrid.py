"""
Generate Hybrid Final Submission: Linear Interpolation for safe interior points, Ridge for dangerous edges
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

df_original = pd.read_csv('dataset.csv')
df = df_original.copy()

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
all_cols = base_aux + option_cols

# 1. Train Ridge Model to learn true covariance for safe extrapolation
imp = IterativeImputer(estimator=Ridge(alpha=0.05), max_iter=50, tol=1e-5, random_state=42)
filled_ridge_matrix = imp.fit_transform(df[all_cols])
filled_ridge = pd.DataFrame(filled_ridge_matrix, columns=all_cols)[option_cols].clip(lower=0.001)

# 2. Apply Hybrid Logic
filled_hybrid = df[option_cols].copy()

for ix, row in df.iterrows():
    if pd.isna(row['datetime']):
        continue
        
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
                    # SAFE INTERPOLATION: Public LB highly rewards this
                    pred = float(f_ce(k))
                    filled_hybrid.at[ix, c] = np.clip(pred, 0.01, 6.0)
                else:
                    # DANGEROUS EXTRAPOLATION: Fallback to Ridge Covariance
                    filled_hybrid.at[ix, c] = filled_ridge.at[ix, c]
    elif len(ce_valid_strikes) == 1:
        for c in ce_cols:
            if pd.isna(row[c]):
                filled_hybrid.at[ix, c] = filled_ridge.at[ix, c]

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
                    # SAFE INTERPOLATION: Public LB highly rewards this
                    pred = float(f_pe(k))
                    filled_hybrid.at[ix, c] = np.clip(pred, 0.01, 6.0)
                else:
                    # DANGEROUS EXTRAPOLATION: Fallback to Ridge Covariance
                    filled_hybrid.at[ix, c] = filled_ridge.at[ix, c]
    elif len(pe_valid_strikes) == 1:
        for c in pe_cols:
            if pd.isna(row[c]):
                filled_hybrid.at[ix, c] = filled_ridge.at[ix, c]

# Global fallback just in case
for c in option_cols:
    if filled_hybrid[c].isna().any():
        filled_hybrid[c] = filled_hybrid[c].fillna(filled_ridge[c])

sub_cols = ['id', 'implied_volatility']
sub_data = []
for ix, row in df.iterrows():
    if pd.isna(row['datetime']):
        continue
    for c in option_cols:
        if pd.isna(row[c]):
            sub_id = f"{row['datetime']}||{c}"
            sub_data.append((sub_id, filled_hybrid.at[ix, c]))

sub_df = pd.DataFrame(sub_data, columns=sub_cols)
sub_df.to_csv('submission_hybrid_ultimate.csv', index=False)
print(f"Saved submission_hybrid_ultimate.csv with {len(sub_df)} rows.")
