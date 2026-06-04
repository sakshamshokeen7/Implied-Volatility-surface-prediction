import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
import time

warnings.filterwarnings('ignore')

print("Loading dataset...")
df_original = pd.read_csv('dataset.csv')
df = df_original.copy()

df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df['time_of_day'] = df['datetime_dt'].dt.hour * 60 + df['datetime_dt'].dt.minute
expiry = pd.to_datetime('2026-01-27 15:30')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600)
df['dte'] = df['dte'].clip(lower=0.001)

option_cols = [c for c in df.columns if c.startswith('NIFTY') and (c.endswith('CE') or c.endswith('PE'))]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])

strikes_ce = np.array([float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols])
strikes_pe = np.array([float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols])

filled_causal = df[option_cols].copy()

# Historical buffers for causal ML training
history_X = []
history_y = []

model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=31, random_state=42)
is_model_trained = False
RETRAIN_FREQ = 100

print("Running Causal Residual Boosting. This takes ~30 seconds...")
start_time = time.time()

for ix in range(len(df)):
    if pd.isna(df.at[ix, 'datetime']):
        continue
        
    row = df.iloc[ix]
    S = row['underlying_price']
    
    # Base features for this row
    row_feats = {
        'time_of_day': row['time_of_day'],
        'dte': row['dte'],
        'S': S,
        'inv_S': 1.0 / S,
        'sq_S': S**2,
        'log_S': np.log(S)
    }
    
    # ----------------------------------------------------
    # Causal Retraining
    # ----------------------------------------------------
    # We only train using data from rows strictly before `ix` 
    # to enforce 100% no lookahead bias.
    if len(history_X) >= 200 and ix % RETRAIN_FREQ == 0:
        # Use a rolling window of the last 15000 residuals to adapt to recent market dynamics 
        # without blowing up memory or training time.
        X_train = np.array(history_X[-15000:])
        y_train = np.array(history_y[-15000:])
        model.fit(X_train, y_train)
        is_model_trained = True

    # ----------------------------------------------------
    # Process CE
    # ----------------------------------------------------
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
            strike = strikes_ce[j]
            linear_pred = float(f_ce(strike))
            
            if pd.isna(row[c]):
                # Predict Missing
                if is_model_trained:
                    x_pred = [row_feats['time_of_day'], row_feats['dte'], strike/S, 1, S, row_feats['inv_S'], row_feats['sq_S'], row_feats['log_S']]
                    residual_pred = model.predict([x_pred])[0]
                else:
                    residual_pred = 0.0
                    
                final_pred = linear_pred + residual_pred
                filled_causal.at[ix, c] = np.clip(final_pred, 0.01, 6.0)
            else:
                # Log True Residual for future training
                true_val = row[c]
                res = true_val - linear_pred
                x_train = [row_feats['time_of_day'], row_feats['dte'], strike/S, 1, S, row_feats['inv_S'], row_feats['sq_S'], row_feats['log_S']]
                history_X.append(x_train)
                history_y.append(res)
    elif len(ce_valid_strikes) == 1:
        for c in ce_cols:
            if pd.isna(row[c]):
                filled_causal.at[ix, c] = ce_valid_ivs[0]

    # ----------------------------------------------------
    # Process PE
    # ----------------------------------------------------
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
            strike = strikes_pe[j]
            linear_pred = float(f_pe(strike))
            
            if pd.isna(row[c]):
                # Predict Missing
                if is_model_trained:
                    x_pred = [row_feats['time_of_day'], row_feats['dte'], strike/S, 0, S, row_feats['inv_S'], row_feats['sq_S'], row_feats['log_S']]
                    residual_pred = model.predict([x_pred])[0]
                else:
                    residual_pred = 0.0
                    
                final_pred = linear_pred + residual_pred
                filled_causal.at[ix, c] = np.clip(final_pred, 0.01, 6.0)
            else:
                # Log True Residual for future training
                true_val = row[c]
                res = true_val - linear_pred
                x_train = [row_feats['time_of_day'], row_feats['dte'], strike/S, 0, S, row_feats['inv_S'], row_feats['sq_S'], row_feats['log_S']]
                history_X.append(x_train)
                history_y.append(res)
    elif len(pe_valid_strikes) == 1:
        for c in pe_cols:
            if pd.isna(row[c]):
                filled_causal.at[ix, c] = pe_valid_ivs[0]
                
    if ix > 0 and ix % 1000 == 0:
        print(f"Processed row {ix}/{len(df)}...")

# Global fallback just in case
for c in option_cols:
    if filled_causal[c].isna().any():
        filled_causal[c] = filled_causal[c].fillna(method='ffill')

print("Preparing submission format...")
sub_cols = ['id', 'implied_volatility']
sub_data = []
for ix, row in df.iterrows():
    if pd.isna(row['datetime']):
        continue
    for c in option_cols:
        if pd.isna(row[c]):
            sub_id = f"{row['datetime']}||{c}"
            sub_data.append((sub_id, filled_causal.at[ix, c]))

sub_df = pd.DataFrame(sub_data, columns=sub_cols)
sub_df.to_csv('submission_causal_residual_boost.csv', index=False)
print(f"Finished in {time.time() - start_time:.2f}s.")
print(f"Saved submission_causal_residual_boost.csv with {len(sub_df)} rows. 100% causal ML.")
