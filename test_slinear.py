import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

df = pd.DataFrame([[np.nan, 0.20, 0.22, np.nan]], columns=[25000, 25100, 25200, 25300])

print("Original:")
print(df)

print("\nPandas slinear:")
print(df.interpolate(method='slinear', axis=1, limit_direction='both'))

print("\nPandas linear:")
print(df.interpolate(method='linear', axis=1, limit_direction='both'))

print("\nPandas index:")
print(df.interpolate(method='index', axis=1, limit_direction='both'))

print("\nScipy interp1d linear extrapolate:")
f = interp1d([25100, 25200], [0.20, 0.22], kind='linear', fill_value='extrapolate')
print([f(25000), f(25100), f(25200), f(25300)])
