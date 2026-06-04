import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

df_original = pd.read_csv('dataset.csv')
df_original = df_original.dropna(subset=['datetime']).reset_index(drop=True)
df = df_original.copy()

feature_cols = [c for c in df_original.columns if c != 'datetime']
option_cols  = [c for c in feature_cols if c != 'underlying_price']

def validate_pchip(df_in, mask_frac=0.15, n_trials=3):
    scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 101 + 13)
        masked = df_in.copy()
        truth = []
        
        # Mask random observed cells
        for c in option_cols:
            obs = df_in.index[df_in[c].notna()].tolist()
            k = max(1, int(len(obs) * mask_frac))
            hide = rng.choice(obs, size=k, replace=False)
            for ix in hide:
                truth.append((ix, c, df_in.at[ix, c]))
                masked.at[ix, c] = np.nan
        
        # Row-by-row PCHIP interpolation
        y_true, y_pred = [], []
        
        strikes = np.array([int(c.replace('NIFTY27JAN26', '').replace('CE', '').replace('PE', '')) for c in option_cols])
        
        for ix, row in masked.iterrows():
            ivs = row[option_cols].values.astype(float)
            valid_idx = ~np.isnan(ivs)
            
            if np.sum(valid_idx) > 2:
                # Interpolate
                valid_strikes = strikes[valid_idx]
                valid_ivs = ivs[valid_idx]
                
                try:
                    interp = PchipInterpolator(valid_strikes, valid_ivs, extrapolate=True)
                    pred_ivs = interp(strikes)
                except:
                    pred_ivs = np.zeros_like(strikes)
            else:
                pred_ivs = np.zeros_like(strikes)
                
            # Collect predictions for truth
            for c_idx, c in enumerate(option_cols):
                # find if this (ix, c) is in truth
                # a bit slow but ok for test
                pass
                
        # optimized collection
        filled = masked.copy()
        for ix in range(len(filled)):
            ivs = filled.loc[ix, option_cols].values.astype(float)
            valid_idx = ~np.isnan(ivs)
            if np.sum(valid_idx) > 1:
                valid_strikes = strikes[valid_idx]
                valid_ivs = ivs[valid_idx]
                # Pchip needs strictly increasing x, they are increasing by default
                try:
                    interp = PchipInterpolator(valid_strikes, valid_ivs, extrapolate=True)
                    filled.loc[ix, option_cols] = interp(strikes)
                except Exception as e:
                    pass
                    
        for ix, c, tv in truth:
            pv = filled.at[ix, c]
            if pd.notna(pv) and pd.notna(tv) and pv > 0:
                y_true.append(tv)
                y_pred.append(pv)
                
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
    
    mu = np.mean(scores)
    print(f"PCHIP Row-by-Row : Mean MSE = {mu:.10f}")
    return mu

if __name__ == '__main__':
    validate_pchip(df.copy())
