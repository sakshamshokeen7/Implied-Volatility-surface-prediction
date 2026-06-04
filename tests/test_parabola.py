import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

df_original = pd.read_csv('dataset.csv')
df_original = df_original.dropna(subset=['datetime']).reset_index(drop=True)
df = df_original.copy()

feature_cols = [c for c in df_original.columns if c != 'datetime']
option_cols  = [c for c in feature_cols if c != 'underlying_price']

def get_strike(c):
    return int(c.replace('NIFTY27JAN26', '').replace('CE', '').replace('PE', ''))

strikes = np.array([get_strike(c) for c in option_cols])

def validate_parabola(df_in, mask_frac=0.15, block_mask=False, n_trials=3):
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
            
            if np.sum(valid_idx) >= 3:
                # Log-moneyness
                x_valid = np.log(strikes[valid_idx] / S)
                y_valid = ivs[valid_idx]
                
                # Fit degree 2 polynomial (parabola)
                coeffs = np.polyfit(x_valid, y_valid, 2)
                poly = np.poly1d(coeffs)
                
                x_all = np.log(strikes / S)
                y_pred_row = poly(x_all)
            else:
                y_pred_row = np.zeros_like(strikes, dtype=float)
                
            for j, c in enumerate(option_cols):
                masked.at[ix, c] = y_pred_row[j]
                
        for ix, c, tv in truth:
            pv = masked.at[ix, c]
            if pd.notna(pv) and pd.notna(tv) and pv > 0:
                y_true.append(tv)
                y_pred.append(pv)
                
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
        
    mu = np.mean(scores)
    mask_type = "Block" if block_mask else "Random"
    print(f"Parabola ({mask_type:6s}) : Mean MSE = {mu:.10f}")
    return mu

if __name__ == '__main__':
    validate_parabola(df.copy(), block_mask=False)
    validate_parabola(df.copy(), block_mask=True)
