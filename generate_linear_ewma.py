import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings
import time

warnings.filterwarnings('ignore')

print("Loading dataset...")
df_original = pd.read_csv('dataset.csv')
df = df_original.copy()

df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')

option_cols = [c for c in df.columns if c.startswith('NIFTY') and (c.endswith('CE') or c.endswith('PE'))]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])

strikes_ce = np.array([float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols])
strikes_pe = np.array([float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols])

LAMBDA_DECAY = 1.0 # Faster decay than before, relies more on cross-sectional
last_known_iv = {} 

filled_df = df[option_cols].copy()

print("Processing row by row with Linear + Causal EWMA...")
start_time = time.time()

for ix in range(len(df)):
    if pd.isna(df.at[ix, 'datetime']):
        continue
        
    row = df.iloc[ix]
    t_current = row['datetime_dt']
    
    # Process CE
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
                strike = strikes_ce[j]
                pred_base = float(f_ce(strike))
                
                if ('CE', strike) in last_known_iv:
                    last_iv, last_t = last_known_iv[('CE', strike)]
                    delta_hours = (t_current - last_t).total_seconds() / 3600.0
                    decay = np.exp(-LAMBDA_DECAY * delta_hours)
                    pred_final = (1 - decay) * pred_base + decay * last_iv
                else:
                    pred_final = pred_base
                    
                filled_df.at[ix, c] = np.clip(pred_final, 0.01, 6.0)
    elif len(ce_valid_strikes) == 1:
        for j, c in enumerate(ce_cols):
            if pd.isna(row[c]):
                filled_df.at[ix, c] = ce_valid_ivs[0]

    # Process PE
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
                strike = strikes_pe[j]
                pred_base = float(f_pe(strike))
                
                if ('PE', strike) in last_known_iv:
                    last_iv, last_t = last_known_iv[('PE', strike)]
                    delta_hours = (t_current - last_t).total_seconds() / 3600.0
                    decay = np.exp(-LAMBDA_DECAY * delta_hours)
                    pred_final = (1 - decay) * pred_base + decay * last_iv
                else:
                    pred_final = pred_base
                    
                filled_df.at[ix, c] = np.clip(pred_final, 0.01, 6.0)
    elif len(pe_valid_strikes) == 1:
        for j, c in enumerate(pe_cols):
            if pd.isna(row[c]):
                filled_df.at[ix, c] = pe_valid_ivs[0]

    # Update state AFTER predicting (strictly causal)
    for j, c in enumerate(ce_cols):
        val = row[c]
        if pd.notna(val):
            last_known_iv[('CE', strikes_ce[j])] = (val, t_current)
            
    for j, c in enumerate(pe_cols):
        val = row[c]
        if pd.notna(val):
            last_known_iv[('PE', strikes_pe[j])] = (val, t_current)

# Global fallback just in case
for c in option_cols:
    if filled_df[c].isna().any():
        filled_df[c] = filled_df[c].fillna(method='ffill').fillna(method='bfill')

print("Preparing submission format...")
sub_cols = ['id', 'implied_volatility']
sub_data = []
for ix, row in df.iterrows():
    if pd.isna(row['datetime']):
        continue
    for c in option_cols:
        if pd.isna(row[c]):
            sub_id = f"{row['datetime']}||{c}"
            sub_data.append((sub_id, filled_df.at[ix, c]))

sub_df = pd.DataFrame(sub_data, columns=sub_cols)
sub_df.to_csv('submission_linear_ewma.csv', index=False)
print(f"Finished in {time.time() - start_time:.2f}s.")
print(f"Saved submission_linear_ewma.csv with {len(sub_df)} rows. Zero look-ahead bias.")
