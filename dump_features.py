"""
Generates and saves OOF residuals and base predictions to speed up ML tuning.
"""
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df_original = pd.read_csv('dataset.csv')
df = df_original.dropna(subset=['datetime']).reset_index(drop=True)
df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df['time_of_day'] = df['datetime_dt'].dt.hour * 60 + df['datetime_dt'].dt.minute
expiry = pd.to_datetime('2026-01-27')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600)

option_cols = [c for c in df.columns if c.endswith('CE') or c.endswith('PE')]
ce_cols = [c for c in option_cols if c.endswith('CE')]
pe_cols = [c for c in option_cols if c.endswith('PE')]

def get_strike(c): return int(c.replace('NIFTY27JAN26', '').replace('CE', '').replace('PE', ''))
ce_strikes = np.array([get_strike(c) for c in ce_cols])
pe_strikes = np.array([get_strike(c) for c in pe_cols])

def interpolate_row(row_idx, row_data, cols, strikes, S, drop_col=None):
    ivs = row_data[cols].values.astype(float)
    x = strikes / S
    valid_idx = ~np.isnan(ivs)
    if drop_col is not None:
        drop_idx = cols.index(drop_col)
        valid_idx[drop_idx] = False
    x_valid = x[valid_idx]
    y_valid = ivs[valid_idx]
    
    if len(x_valid) > 1:
        sort_idx = np.argsort(x_valid)
        x_valid = x_valid[sort_idx]
        y_valid = y_valid[sort_idx]
        
        interp = PchipInterpolator(x_valid, y_valid)
        res = np.zeros_like(x)
        for i, xv in enumerate(x):
            if xv <= x_valid[0]: res[i] = y_valid[0]
            elif xv >= x_valid[-1]: res[i] = y_valid[-1]
            else: res[i] = interp(xv)
        return res
    elif len(x_valid) == 1:
        return np.full_like(x, y_valid[0])
    else:
        return np.full_like(x, np.nan)

print("Generating OOF data...")
oof_data = []
for ix, row in df.iterrows():
    S = row['underlying_price']
    for j, c in enumerate(pe_cols):
        true_val = row[c]
        if pd.notna(true_val):
            preds = interpolate_row(ix, row, pe_cols, pe_strikes, S, drop_col=c)
            pred_val = preds[j]
            if not np.isnan(pred_val):
                oof_data.append({
                    'time_of_day': row['time_of_day'], 'dte': row['dte'],
                    'S': S, 'inv_S': 1.0/S, 'sq_S': S**2, 'log_S': np.log(S),
                    'strike': pe_strikes[j], 'moneyness': pe_strikes[j] / S, 'is_ce': 0,
                    'pchip_iv': pred_val, 'true_iv': true_val, 'residual': true_val - pred_val
                })
    for j, c in enumerate(ce_cols):
        true_val = row[c]
        if pd.notna(true_val):
            preds = interpolate_row(ix, row, ce_cols, ce_strikes, S, drop_col=c)
            pred_val = preds[j]
            if not np.isnan(pred_val):
                oof_data.append({
                    'time_of_day': row['time_of_day'], 'dte': row['dte'],
                    'S': S, 'inv_S': 1.0/S, 'sq_S': S**2, 'log_S': np.log(S),
                    'strike': ce_strikes[j], 'moneyness': ce_strikes[j] / S, 'is_ce': 1,
                    'pchip_iv': pred_val, 'true_iv': true_val, 'residual': true_val - pred_val
                })

oof_df = pd.DataFrame(oof_data)
oof_df.to_csv('oof_data.csv', index=False)
print(f"Saved {len(oof_df)} OOF rows.")

print("Generating base predictions for missing values...")
base_preds = []
for ix, row in df.iterrows():
    S = row['underlying_price']
    pe_preds = interpolate_row(ix, row, pe_cols, pe_strikes, S, drop_col=None)
    for j, c in enumerate(pe_cols):
        if pd.isna(row[c]):
            base_preds.append({
                'row_idx': ix, 'col': c, 'base_iv': pe_preds[j],
                'time_of_day': row['time_of_day'], 'dte': row['dte'],
                'S': S, 'inv_S': 1.0/S, 'sq_S': S**2, 'log_S': np.log(S),
                'strike': pe_strikes[j], 'moneyness': pe_strikes[j] / S, 'is_ce': 0
            })
    
    ce_preds = interpolate_row(ix, row, ce_cols, ce_strikes, S, drop_col=None)
    for j, c in enumerate(ce_cols):
        if pd.isna(row[c]):
            base_preds.append({
                'row_idx': ix, 'col': c, 'base_iv': ce_preds[j],
                'time_of_day': row['time_of_day'], 'dte': row['dte'],
                'S': S, 'inv_S': 1.0/S, 'sq_S': S**2, 'log_S': np.log(S),
                'strike': ce_strikes[j], 'moneyness': ce_strikes[j] / S, 'is_ce': 1
            })

pd.DataFrame(base_preds).to_csv('base_preds.csv', index=False)
print("Saved base predictions.")
