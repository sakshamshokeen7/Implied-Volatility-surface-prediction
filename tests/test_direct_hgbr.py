"""
Direct Regression on flattened dataset.
Treat every valid IV cell as a training sample.
Features: S, time_of_day, dte, strike, is_ce, moneyness.
Target: IV.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df_original = pd.read_csv('dataset.csv')
df = df_original.dropna(subset=['datetime']).reset_index(drop=True)

df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df['time_of_day'] = df['datetime_dt'].dt.hour * 60 + df['datetime_dt'].dt.minute
expiry = pd.to_datetime('2026-01-27 15:30')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600)
df['dte'] = df['dte'].clip(lower=0.001)

option_cols = [c for c in df_original.columns if c.startswith('NIFTY')]

def get_strike(c): return float(c.replace('NIFTY27JAN26','').replace('CE','').replace('PE',''))
def get_is_ce(c): return 1.0 if c.endswith('CE') else 0.0

strikes = {c: get_strike(c) for c in option_cols}
is_ce = {c: get_is_ce(c) for c in option_cols}

def validate(n_trials=3):
    scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 101 + 13)
        masked = df.copy()
        
        # Hide 15% of cells
        truth = []
        for c in option_cols:
            obs = df.index[df[c].notna()].tolist()
            k = max(1, int(len(obs) * 0.15))
            hide = rng.choice(obs, size=k, replace=False)
            for ix in hide:
                truth.append((ix, c, df.at[ix, c]))
                masked.at[ix, c] = np.nan
        
        # Flatten training data
        train_rows = []
        for ix, row in masked.iterrows():
            S = row['underlying_price']
            t = row['time_of_day']
            dte = row['dte']
            for c in option_cols:
                v = row[c]
                if pd.notna(v):
                    train_rows.append([S, t, dte, strikes[c], strikes[c]/S, is_ce[c], v])
                    
        train_df = pd.DataFrame(train_rows, columns=['S', 'time', 'dte', 'strike', 'moneyness', 'is_ce', 'iv'])
        X_train = train_df[['S', 'time', 'dte', 'strike', 'moneyness', 'is_ce']]
        y_train = train_df['iv']
        
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=63, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict hidden
        y_true, y_pred = [], []
        for ix, c, tv in truth:
            S = df.at[ix, 'underlying_price']
            t = df.at[ix, 'time_of_day']
            dte = df.at[ix, 'dte']
            X_test = pd.DataFrame([[S, t, dte, strikes[c], strikes[c]/S, is_ce[c]]], 
                                  columns=['S', 'time', 'dte', 'strike', 'moneyness', 'is_ce'])
            pv = max(0.001, model.predict(X_test)[0])
            y_true.append(tv)
            y_pred.append(pv)
            
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
        print(f"Trial {trial} MSE: {mse:.10f}")
        
    mu = np.mean(scores)
    print(f"Direct Flattened HGBR : {mu:.10f}")
    return mu

print("Testing Direct Flattened Regression...")
validate()
