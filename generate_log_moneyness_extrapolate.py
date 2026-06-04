"""
Generate submission using Robust Log-Moneyness Linear Extrapolation
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

print("Generating predictions using Log-Moneyness Linear Extrapolation...")

for ix, row in df.iterrows():
    if pd.isna(row['datetime']):
        continue
        
    S = row['underlying_price']
    
    # ------------------- CE -------------------
    valid_ce_strikes = []
    valid_ce_ivs = []
    for j, c in enumerate(ce_cols):
        val = row[c]
        if pd.notna(val):
            valid_ce_strikes.append(strikes_ce[j])
            valid_ce_ivs.append(val)
            
    if len(valid_ce_strikes) >= 2:
        valid_ce_log_m = np.log(np.array(valid_ce_strikes) / S)
        f_ce = interp1d(valid_ce_log_m, valid_ce_ivs, kind='linear', fill_value='extrapolate')
        for j, c in enumerate(ce_cols):
            if pd.isna(row[c]):
                log_m = np.log(strikes_ce[j] / S)
                pred = float(f_ce(log_m))
                filled.at[ix, c] = np.clip(pred, 0.01, 6.0)
    elif len(valid_ce_strikes) == 1:
        for c in ce_cols:
            if pd.isna(row[c]):
                filled.at[ix, c] = valid_ce_ivs[0]

    # ------------------- PE -------------------
    valid_pe_strikes = []
    valid_pe_ivs = []
    for j, c in enumerate(pe_cols):
        val = row[c]
        if pd.notna(val):
            valid_pe_strikes.append(strikes_pe[j])
            valid_pe_ivs.append(val)
            
    if len(valid_pe_strikes) >= 2:
        valid_pe_log_m = np.log(np.array(valid_pe_strikes) / S)
        f_pe = interp1d(valid_pe_log_m, valid_pe_ivs, kind='linear', fill_value='extrapolate')
        for j, c in enumerate(pe_cols):
            if pd.isna(row[c]):
                log_m = np.log(strikes_pe[j] / S)
                pred = float(f_pe(log_m))
                filled.at[ix, c] = np.clip(pred, 0.01, 6.0)
    elif len(valid_pe_strikes) == 1:
        for c in pe_cols:
            if pd.isna(row[c]):
                filled.at[ix, c] = valid_pe_ivs[0]
                
    if ix > 0 and ix % 1000 == 0:
        print(f"Processed row {ix}/{len(df)}...")

# Final fallback for completely empty rows (if any)
for c in option_cols:
    if filled[c].isna().any():
        filled[c] = filled[c].fillna(method='ffill')

print("Preparing submission format...")
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
sub_df.to_csv('submission_log_moneyness_extrapolate.csv', index=False)
print(f"Finished generating submission_log_moneyness_extrapolate.csv with {len(sub_df)} rows.")
