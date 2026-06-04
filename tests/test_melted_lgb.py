import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
import time

warnings.filterwarnings('ignore')

df_original = pd.read_csv('dataset.csv')
df_original = df_original.dropna(subset=['datetime']).reset_index(drop=True)
df = df_original.copy()

df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df['time_of_day'] = df['datetime_dt'].dt.hour * 60 + df['datetime_dt'].dt.minute
expiry = pd.to_datetime('2026-01-27')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600)

option_cols = [c for c in df.columns if c.endswith('CE') or c.endswith('PE')]

def get_strike(c):
    return int(c.replace('NIFTY27JAN26', '').replace('CE', '').replace('PE', ''))

def validate_melted(df_in, mask_frac=0.15, n_trials=3):
    scores = []
    
    # Melt the data into a panel format
    id_vars = ['datetime_dt', 'time_of_day', 'dte', 'underlying_price']
    melted = df_in.melt(id_vars=id_vars, value_vars=option_cols, var_name='option', value_name='iv')
    
    melted['strike'] = melted['option'].apply(get_strike)
    melted['is_ce'] = melted['option'].str.endswith('CE').astype(int)
    melted['moneyness'] = melted['strike'] / melted['underlying_price']
    melted['log_moneyness'] = np.log(melted['moneyness'])
    
    # Add time-series features (e.g. lag IV of the ATM option, or rolling underlying)
    # For now, just use the row features
    features = ['time_of_day', 'dte', 'underlying_price', 'strike', 'moneyness', 'log_moneyness', 'is_ce']
    
    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 101 + 13)
        
        # We only consider rows where IV is observed
        observed = melted[melted['iv'].notna()].copy()
        
        # Mask random 15%
        mask_idx = rng.choice(observed.index, size=int(len(observed) * mask_frac), replace=False)
        train_data = observed.drop(mask_idx)
        test_data = observed.loc[mask_idx]
        
        X_train = train_data[features]
        y_train = train_data['iv']
        
        X_test = test_data[features]
        y_test = test_data['iv']
        
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        scores.append(mse)
        
    mu = np.mean(scores)
    print(f"Melted LGBM : Mean MSE = {mu:.10f}")
    return mu

if __name__ == '__main__':
    validate_melted(df.copy())
