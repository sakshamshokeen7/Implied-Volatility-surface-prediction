"""
Robust local CV for the notebook.py pipeline (PCHIP + HGBR residual).
"""
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df_original = pd.read_csv('dataset.csv')
df = df_original.dropna(subset=['datetime']).reset_index(drop=True)
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
                if xv <= x_valid[0]: res[i] = y_valid[0]
                elif xv >= x_valid[-1]: res[i] = y_valid[-1]
                else: res[i] = interp(xv)
            return res
        return pchip_extrap(x)
    elif len(x_valid) == 1:
        return np.full_like(x, y_valid[0])
    else:
        return np.full_like(x, np.nan)

def validate_notebook_pipeline(n_trials=2):
    scores = []
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

        # Generate OOF data for training
        oof_data = []
        for ix, row in masked.iterrows():
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
                            'moneyness': pe_strikes[j] / S, 'is_ce': 0,
                            'residual': true_val - pred_val
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
                            'moneyness': ce_strikes[j] / S, 'is_ce': 1,
                            'residual': true_val - pred_val
                        })

        oof_df = pd.DataFrame(oof_data)
        features = ['time_of_day', 'dte', 'moneyness', 'is_ce', 'S', 'inv_S', 'sq_S', 'log_S']
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=31, random_state=42)
        model.fit(oof_df[features], oof_df['residual'])

        filled = masked[option_cols].copy()
        for ix, row in masked.iterrows():
            S = row['underlying_price']
            pe_preds = interpolate_row(ix, row, pe_cols, pe_strikes, S, drop_col=None)
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]):
                    base_iv = pe_preds[j]
                    if not np.isnan(base_iv):
                        X_pred = pd.DataFrame([{'time_of_day': row['time_of_day'], 'dte': row['dte'], 'moneyness': pe_strikes[j]/S, 'is_ce': 0, 'S': S, 'inv_S': 1.0/S, 'sq_S': S**2, 'log_S': np.log(S)}])
                        filled.at[ix, c] = max(0.001, base_iv + model.predict(X_pred)[0])
            ce_preds = interpolate_row(ix, row, ce_cols, ce_strikes, S, drop_col=None)
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]):
                    base_iv = ce_preds[j]
                    if not np.isnan(base_iv):
                        X_pred = pd.DataFrame([{'time_of_day': row['time_of_day'], 'dte': row['dte'], 'moneyness': ce_strikes[j]/S, 'is_ce': 1, 'S': S, 'inv_S': 1.0/S, 'sq_S': S**2, 'log_S': np.log(S)}])
                        filled.at[ix, c] = max(0.001, base_iv + model.predict(X_pred)[0])

        y_true, y_pred = [], []
        for ix, c, tv in truth:
            pv = filled.at[ix, c]
            if pd.notna(pv) and pd.notna(tv):
                y_true.append(tv)
                y_pred.append(pv)
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
    
    print(f"Notebook Pipeline CV: {np.mean(scores):.10f}")

print("Testing notebook.py baseline...")
validate_notebook_pipeline()
