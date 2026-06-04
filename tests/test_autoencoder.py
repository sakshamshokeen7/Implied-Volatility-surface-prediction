"""
Autoencoder Imputation for IV Surface.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df_original = pd.read_csv('dataset.csv')
df = df_original.dropna(subset=['datetime']).reset_index(drop=True)

option_cols = [c for c in df_original.columns if c.startswith('NIFTY')]
aux_cols = ['underlying_price']

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

def validate(n_trials=3):
    scores = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        
        all_cols = aux_cols + option_cols
        X = masked[all_cols].copy()
        
        # Initial fill
        mask = X.isna().values
        X_filled = X.fillna(X.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)
        
        model = Autoencoder(len(all_cols)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        mask_tensor = torch.BoolTensor(mask).to(device)
        
        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            out = model(X_tensor)
            
            # Only compute loss on observed values
            loss = criterion(out[~mask_tensor], X_tensor[~mask_tensor])
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                X_tensor[mask_tensor] = out[mask_tensor]
                
        model.eval()
        with torch.no_grad():
            final_out = model(X_tensor).cpu().numpy()
            
        final_unscaled = scaler.inverse_transform(final_out)
        filled_df = pd.DataFrame(final_unscaled, columns=all_cols)
        
        y_true, y_pred = [], []
        for ix, c, tv in truth:
            pv = max(0.001, filled_df.at[ix, c])
            y_true.append(tv)
            y_pred.append(pv)
            
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
        print(f"Trial {trial} MSE: {mse:.10f}")
        
    mu = np.mean(scores)
    print(f"Autoencoder Imputer : {mu:.10f}")
    return mu

print("Testing Autoencoder Imputer...")
validate()
