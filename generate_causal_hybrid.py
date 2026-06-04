"""
Generate Final Submission (Causal Hybrid Expanding Window)
Strictly enforces NO look-ahead bias while maintaining Ridge extrapolation power.
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
import warnings
import time
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

filled_causal = df[option_cols].copy()

print("Running causal expanding-window Hybrid Imputer. This may take ~2 minutes...")
start_time = time.time()

for ix in range(len(df)):
    if pd.isna(df.at[ix, 'datetime']):
        continue
        
    row = df.iloc[ix]
    
    # 1. Causal Ridge Prediction (Expanding Window)
    # Only use rows 0 to ix (inclusive for fitting? NO, inclusive of ix means we see ix's known values,
    # which is fine! But we shouldn't see future rows. Using df.iloc[:ix+1] is perfectly causal because
    # row ix is the CURRENT time. We only see the known values at time t to predict missing values at time t).
    if ix >= 10:
        historical_data = df.iloc[:ix+1][all_cols].copy()
        imp = IterativeImputer(estimator=Ridge(alpha=0.05), max_iter=30, tol=1e-5, random_state=42)
        # We only care about the imputed result of the VERY LAST row (current time t)
        imputed_history = imp.fit_transform(historical_data)
        ridge_pred = imputed_history[-1]
    else:
        # Fallback for very first few rows where Ridge has no history
        ridge_pred = np.zeros(len(all_cols))
        
    ridge_dict = {col: max(0.01, ridge_pred[idx]) for idx, col in enumerate(all_cols)}
    
    # 2. Apply Hybrid Logic for the current row
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
                    # SAFE INTERPOLATION
                    pred = float(f_ce(k))
                    filled_causal.at[ix, c] = np.clip(pred, 0.01, 6.0)
                else:
                    # DANGEROUS EXTRAPOLATION: Fallback to Causal Ridge
                    if ix >= 10:
                        filled_causal.at[ix, c] = ridge_dict[c]
                    else:
                        pred = float(f_ce(k))
                        filled_causal.at[ix, c] = np.clip(pred, 0.01, 6.0)
    elif len(ce_valid_strikes) == 1:
        for c in ce_cols:
            if pd.isna(row[c]):
                if ix >= 10:
                    filled_causal.at[ix, c] = ridge_dict[c]
                else:
                    filled_causal.at[ix, c] = ce_valid_ivs[0]

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
                    # SAFE INTERPOLATION
                    pred = float(f_pe(k))
                    filled_causal.at[ix, c] = np.clip(pred, 0.01, 6.0)
                else:
                    # DANGEROUS EXTRAPOLATION: Fallback to Causal Ridge
                    if ix >= 10:
                        filled_causal.at[ix, c] = ridge_dict[c]
                    else:
                        pred = float(f_pe(k))
                        filled_causal.at[ix, c] = np.clip(pred, 0.01, 6.0)
    elif len(pe_valid_strikes) == 1:
        for c in pe_cols:
            if pd.isna(row[c]):
                if ix >= 10:
                    filled_causal.at[ix, c] = ridge_dict[c]
                else:
                    filled_causal.at[ix, c] = pe_valid_ivs[0]
                    
    if ix % 100 == 0:
        print(f"Processed row {ix}/{len(df)}...")

# Global fallback just in case
for c in option_cols:
    if filled_causal[c].isna().any():
        filled_causal[c] = filled_causal[c].fillna(method='ffill')

sub_cols = ['id', 'implied_volatility']
sub_data = []
for ix, row in df.iterrows():
    if pd.isna(row['datetime']):
        continue
    for c in option_cols:
        if pd.isna(row[c]):
            sub_id = f"{row['datetime']}||{c}"
            sub_data.append((sub_id, filled_causal.at[ix, c]))

sub_df = pd.DataFrame(sub_data, columns=sub_cols)
sub_df.to_csv('submission_causal_hybrid.csv', index=False)
print(f"Finished in {time.time() - start_time:.2f}s.")
print(f"Saved submission_causal_hybrid.csv with {len(sub_df)} rows. 100% look-ahead bias free.")
