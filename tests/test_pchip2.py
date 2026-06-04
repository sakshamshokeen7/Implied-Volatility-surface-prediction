import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, UnivariateSpline, CubicSpline
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

df_original = pd.read_csv('dataset.csv')
df_original = df_original.dropna(subset=['datetime']).reset_index(drop=True)
df = df_original.copy()

feature_cols = [c for c in df_original.columns if c != 'datetime']
option_cols  = [c for c in feature_cols if c != 'underlying_price']

def get_strike(c):
    return int(c.replace('NIFTY27JAN26', '').replace('CE', '').replace('PE', ''))

strikes = np.array([get_strike(c) for c in option_cols])

def validate_row_by_row(df_in, method='pchip', mask_frac=0.15, block_mask=False, n_trials=3):
    scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 101 + 13)
        masked = df_in.copy()
        truth = []
        
        for ix, row in masked.iterrows():
            obs = [j for j, c in enumerate(option_cols) if pd.notna(row[c])]
            
            if len(obs) < 4:
                continue
                
            if block_mask:
                k = max(1, int(len(obs) * mask_frac))
                # pick a random starting point for the block
                start_idx = rng.randint(0, len(obs) - k + 1)
                hide_idx = obs[start_idx : start_idx + k]
            else:
                k = max(1, int(len(obs) * mask_frac))
                hide_idx = rng.choice(obs, size=k, replace=False)
                
            for j in hide_idx:
                c = option_cols[j]
                truth.append((ix, c, df_in.at[ix, c]))
                masked.at[ix, c] = np.nan
        
        y_true, y_pred = [], []
        
        for ix, row in masked.iterrows():
            ivs = row[option_cols].values.astype(float)
            S = row['underlying_price']
            
            valid_idx = ~np.isnan(ivs)
            
            if np.sum(valid_idx) > 3:
                x_valid = strikes[valid_idx] / S  # Moneyness space
                y_valid = ivs[valid_idx]
                
                # Sort x to be strictly increasing (should be already, but just in case)
                sort_idx = np.argsort(x_valid)
                x_valid = x_valid[sort_idx]
                y_valid = y_valid[sort_idx]
                
                x_all = strikes / S
                
                try:
                    if method == 'pchip':
                        interp = PchipInterpolator(x_valid, y_valid, extrapolate=True)
                        y_pred_row = interp(x_all)
                    elif method == 'spline':
                        interp = UnivariateSpline(x_valid, y_valid, k=3, s=0.0001, ext=0)
                        y_pred_row = interp(x_all)
                    elif method == 'cubic':
                        interp = CubicSpline(x_valid, y_valid, bc_type='natural', extrapolate=True)
                        y_pred_row = interp(x_all)
                except:
                    y_pred_row = np.zeros_like(x_all)
            else:
                y_pred_row = np.zeros_like(strikes, dtype=float)
                
            for j, c in enumerate(option_cols):
                pv = y_pred_row[j]
                masked.at[ix, c] = pv
                
        for ix, c, tv in truth:
            pv = masked.at[ix, c]
            if pd.notna(pv) and pd.notna(tv) and pv > 0:
                y_true.append(tv)
                y_pred.append(pv)
                
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
        
    mu = np.mean(scores)
    mask_type = "Block" if block_mask else "Random"
    print(f"{method:10s} ({mask_type:6s}) : Mean MSE = {mu:.10f}")
    return mu

if __name__ == '__main__':
    for method in ['pchip', 'spline', 'cubic']:
        validate_row_by_row(df.copy(), method=method, block_mask=False)
        validate_row_by_row(df.copy(), method=method, block_mask=True)
