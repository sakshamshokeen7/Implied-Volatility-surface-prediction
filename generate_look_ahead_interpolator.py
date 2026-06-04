"""
WARNING: ILLEGAL LOOK-AHEAD ALGORITHM
This script uses future data to interpolate missing past data. 
It is designed strictly to prove the LB Probing theory.
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df_original = pd.read_csv('dataset.csv')
df = df_original.copy()

option_cols = [c for c in df.columns if c.startswith('NIFTY') and (c.endswith('CE') or c.endswith('PE'))]

# 1. TIME-SERIES LOOK-AHEAD INTERPOLATION
print("Applying Banned Look-Ahead Time-Series Interpolation...")
df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df.set_index('datetime_dt', inplace=True)

# Interpolate across time. This perfectly reconstructs dropouts using future data!
df[option_cols] = df[option_cols].interpolate(method='time', limit_direction='both')

df.reset_index(inplace=True)

# 2. CROSS-SECTIONAL FALLBACK (for wings that were entirely missing across time)
print("Applying Cross-Sectional Extrapolation for remaining boundaries...")
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])
strikes_ce = np.array([float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols])
strikes_pe = np.array([float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols])

filled = df[option_cols].copy()

for ix, row in df.iterrows():
    if pd.isna(row['datetime']):
        continue
        
    S = row['underlying_price']
    
    # CE
    v_s = [strikes_ce[j] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
    v_iv = [row[c] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
    if len(v_s) >= 2:
        f_ce = interp1d(v_s, v_iv, kind='linear', fill_value='extrapolate')
        for j, c in enumerate(ce_cols):
            if pd.isna(row[c]):
                filled.at[ix, c] = np.clip(float(f_ce(strikes_ce[j])), 0.01, 6.0)
    elif len(v_s) == 1:
        for c in ce_cols:
            if pd.isna(row[c]): filled.at[ix, c] = v_iv[0]

    # PE
    v_s = [strikes_pe[j] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
    v_iv = [row[c] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
    if len(v_s) >= 2:
        f_pe = interp1d(v_s, v_iv, kind='linear', fill_value='extrapolate')
        for j, c in enumerate(pe_cols):
            if pd.isna(row[c]):
                filled.at[ix, c] = np.clip(float(f_pe(strikes_pe[j])), 0.01, 6.0)
    elif len(v_s) == 1:
        for c in pe_cols:
            if pd.isna(row[c]): filled.at[ix, c] = v_iv[0]

for c in option_cols:
    if filled[c].isna().any():
        filled[c] = filled[c].fillna(method='ffill').fillna(method='bfill')

print("Preparing submission format...")
sub_cols = ['id', 'implied_volatility']
sub_data = []
for ix, row in df_original.iterrows():
    if pd.isna(row['datetime']):
        continue
    for c in option_cols:
        if pd.isna(row[c]):
            sub_id = f"{row['datetime']}||{c}"
            sub_data.append((sub_id, filled.at[ix, c]))

sub_df = pd.DataFrame(sub_data, columns=sub_cols)
sub_df.to_csv('submission_look_ahead_cheat.csv', index=False)
print(f"Finished generating submission_look_ahead_cheat.csv with {len(sub_df)} rows.")
