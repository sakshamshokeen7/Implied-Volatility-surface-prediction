import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
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

# Lambda for EWMA decay (per hour)
# e.g., 0.5 means weight of past observation halves every ~1.4 hours
LAMBDA_DECAY = 0.5 
last_known_iv = {} # (is_ce, strike) -> (iv, last_time)

filled_df = df[option_cols].copy()

print("Processing row by row with Causal PCHIP + EWMA...")
start_time = time.time()

for ix in range(len(df)):
    if pd.isna(df.at[ix, 'datetime']):
        continue
        
    row = df.iloc[ix]
    S = row['underlying_price']
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
        # Convert to Log-Moneyness
        k_valid = np.log(np.array(ce_valid_strikes) / S)
        y_valid = np.array(ce_valid_ivs)
        
        # Sort just in case
        sort_idx = np.argsort(k_valid)
        k_valid = k_valid[sort_idx]
        y_valid = y_valid[sort_idx]
        
        pchip = PchipInterpolator(k_valid, y_valid)
        
        for j, c in enumerate(ce_cols):
            if pd.isna(row[c]):
                strike = strikes_ce[j]
                k_target = np.log(strike / S)
                
                # Flat Extrapolation
                if k_target <= k_valid[0]:
                    iv_base = y_valid[0]
                elif k_target >= k_valid[-1]:
                    iv_base = y_valid[-1]
                else:
                    iv_base = float(pchip(k_target))
                
                # Causal EWMA Blending
                if ('CE', strike) in last_known_iv:
                    last_iv, last_t = last_known_iv[('CE', strike)]
                    delta_hours = (t_current - last_t).total_seconds() / 3600.0
                    decay = np.exp(-LAMBDA_DECAY * delta_hours)
                    iv_final = (1 - decay) * iv_base + decay * last_iv
                else:
                    iv_final = iv_base
                    
                filled_df.at[ix, c] = max(0.01, iv_final)
                
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
        k_valid = np.log(np.array(pe_valid_strikes) / S)
        y_valid = np.array(pe_valid_ivs)
        
        sort_idx = np.argsort(k_valid)
        k_valid = k_valid[sort_idx]
        y_valid = y_valid[sort_idx]
        
        pchip = PchipInterpolator(k_valid, y_valid)
        
        for j, c in enumerate(pe_cols):
            if pd.isna(row[c]):
                strike = strikes_pe[j]
                k_target = np.log(strike / S)
                
                # Flat Extrapolation
                if k_target <= k_valid[0]:
                    iv_base = y_valid[0]
                elif k_target >= k_valid[-1]:
                    iv_base = y_valid[-1]
                else:
                    iv_base = float(pchip(k_target))
                
                # Causal EWMA Blending
                if ('PE', strike) in last_known_iv:
                    last_iv, last_t = last_known_iv[('PE', strike)]
                    delta_hours = (t_current - last_t).total_seconds() / 3600.0
                    decay = np.exp(-LAMBDA_DECAY * delta_hours)
                    iv_final = (1 - decay) * iv_base + decay * last_iv
                else:
                    iv_final = iv_base
                    
                filled_df.at[ix, c] = max(0.01, iv_final)
                
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
        filled_df[c] = filled_df[c].fillna(method='ffill')

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
sub_df.to_csv('submission_pchip_ewma.csv', index=False)
print(f"Finished in {time.time() - start_time:.2f}s.")
print(f"Saved submission_pchip_ewma.csv with {len(sub_df)} rows. Zero look-ahead bias.")
