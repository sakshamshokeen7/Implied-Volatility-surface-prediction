import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
import time

warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('dataset.csv')
df = df.dropna(subset=['datetime']).reset_index(drop=True)

df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df['time_of_day'] = df['datetime_dt'].dt.hour * 60 + df['datetime_dt'].dt.minute
expiry = pd.to_datetime('2026-01-27')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600)

option_cols = [c for c in df.columns if c.endswith('CE') or c.endswith('PE')]
ce_cols = [c for c in option_cols if c.endswith('CE')]
pe_cols = [c for c in option_cols if c.endswith('PE')]

def get_strike(c):
    return int(c.replace('NIFTY27JAN26', '').replace('CE', '').replace('PE', ''))

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
        
        def pchip_extrap(xi):
            interp = PchipInterpolator(x_valid, y_valid)
            res = np.zeros_like(xi)
            for i, xv in enumerate(xi):
                if xv <= x_valid[0]:
                    res[i] = y_valid[0]
                elif xv >= x_valid[-1]:
                    res[i] = y_valid[-1]
                else:
                    res[i] = interp(xv)
            return res
            
        return pchip_extrap(x)
    elif len(x_valid) == 1:
        return np.full_like(x, y_valid[0])
    else:
        return np.full_like(x, np.nan)

print("Running LOO PCHIP to generate OOF residuals...")
oof_data = []
t0 = time.time()
for ix, row in df.iterrows():
    S = row['underlying_price']
    
    for j, c in enumerate(pe_cols):
        true_val = row[c]
        if pd.notna(true_val):
            preds = interpolate_row(ix, row, pe_cols, pe_strikes, S, drop_col=c)
            pred_val = preds[j]
            if not np.isnan(pred_val):
                oof_data.append({
                    'datetime': row['datetime_dt'],
                    'time_of_day': row['time_of_day'],
                    'dte': row['dte'],
                    'S': S,
                    'strike': pe_strikes[j],
                    'moneyness': pe_strikes[j] / S,
                    'is_ce': 0,
                    'pchip_iv': pred_val,
                    'true_iv': true_val,
                    'residual': true_val - pred_val
                })
                
    for j, c in enumerate(ce_cols):
        true_val = row[c]
        if pd.notna(true_val):
            preds = interpolate_row(ix, row, ce_cols, ce_strikes, S, drop_col=c)
            pred_val = preds[j]
            if not np.isnan(pred_val):
                oof_data.append({
                    'datetime': row['datetime_dt'],
                    'time_of_day': row['time_of_day'],
                    'dte': row['dte'],
                    'S': S,
                    'strike': ce_strikes[j],
                    'moneyness': ce_strikes[j] / S,
                    'is_ce': 1,
                    'pchip_iv': pred_val,
                    'true_iv': true_val,
                    'residual': true_val - pred_val
                })

oof_df = pd.DataFrame(oof_data)
print(f"Generated {len(oof_df)} OOF samples in {time.time()-t0:.1f}s")
print(f"Base PCHIP LOO MSE: {mean_squared_error(oof_df['true_iv'], oof_df['pchip_iv']):.10f}")

print("Training HistGradientBoostingRegressor (Time-Series Split)...")
oof_df = oof_df.sort_values('datetime')
features = ['time_of_day', 'dte', 'moneyness', 'is_ce']
X = oof_df[features]
y = oof_df['residual']

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
lgb_mses = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    corrected_iv = oof_df.iloc[test_index]['pchip_iv'] + preds
    mse = mean_squared_error(oof_df.iloc[test_index]['true_iv'], corrected_iv)
    lgb_mses.append(mse)

print(f"HGB Corrected LOO MSE (CV): {np.mean(lgb_mses):.10f}")
