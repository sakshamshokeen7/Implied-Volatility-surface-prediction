"""
Generate submission using Deterministic Cross-Sectional Linear Extrapolation
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

df_original = pd.read_csv('dataset.csv')
df = df_original.copy()

option_cols = [c for c in df_original.columns if c.startswith('NIFTY')]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])

strikes_ce = np.array([float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols])
strikes_pe = np.array([float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols])

filled = df[option_cols].copy()

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
                
# Global fallback
for c in option_cols:
    if filled[c].isna().any():
        filled[c] = filled[c].fillna(filled[c].mean())

sub_cols = ['id', 'implied_volatility']
sub_data = []
for ix, row in df.iterrows():
    if pd.isna(row['datetime']):
        continue
    for c in option_cols:
        if pd.isna(row[c]):
            sub_id = f"{row['datetime']}||{c}"
            sub_data.append((sub_id, filled.at[ix, c]))

sub_df = pd.DataFrame(sub_data, columns=sub_cols)
sub_df.to_csv('submission_linear_extrapolate.csv', index=False)
print(f"Saved submission_linear_extrapolate.csv with {len(sub_df)} rows.")
