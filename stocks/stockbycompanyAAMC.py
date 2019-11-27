import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('STOCK-PRICES.csv')

dfaamc = df.loc[df['ticker'] == 'AAMC']
print(dfaamc.head())