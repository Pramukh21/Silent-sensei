import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('STOCK-PRICES.csv')
dfaan = df.loc[df['ticker'] == 'AAN']
print(dfaan.head())