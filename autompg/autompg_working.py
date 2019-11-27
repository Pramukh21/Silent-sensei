import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

df = pd.read_csv('autompg2.csv')
print(df.head())
df.info()
print(df.describe())
df.columns

co = df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration']].corr()
print("\n The corelation between each variables are given below \n",co)

avgmpg = df['mpg'].mean()
print("\n The average miles per gallon of all the cars in the data is: ",avgmpg)

avgcy = df['cylinders'].mean()
print("\n The average number of cylinders is : ",avgcy)

avgdisp = df['displacement'].mean()
print("\n The average of displacement is : ",avgdisp)

avgwe = df['weight'].mean()
print("\n The average weight is : ",avgwe)

avgacc = df['acceleration'].mean()
print("\n The average acceleration is : ",avgacc)

stdmpg = df['mpg'].std()
print("\n The Standard deviation for miles per gallon is : ",stdmpg)

stdcyl = df['cylinders'].std()
print("\n The Standard deviation for cylinders is : ",stdcyl)

stddisp = df['displacement'].std()
print("\n The Standard deviation for displacement is : ",stddisp)

stdwe = df['weight'].std()
print("\n The Standard deviation for weight is : ",stdwe)

stdacc = df['acceleration'].std()
print("\n The Standard deviation for acceleration is : ",stdacc)

varmpg = df['mpg'].var()
print("\n The variance of miles per gallon is : ",varmpg)

varcyl = df['cylinders'].var()
print("\n The variance of cylinders is : ",varcyl)

vardisp = df['displacement'].var()
print("\n The variance of displacement is : ", vardisp)

varwe = df['weight'].var()
print("\n The variance of weight is : ",varwe)

varacc = df['acceleration'].var()
print("\n The variance of acceleration is : ",varacc)

minmpg = df['mpg'].min()
print("\n The minimum miles per gallon is : ",minmpg)

mincyl = df['cylinders'].min()
print("\n The minimum number of cylinders is : ",mincyl)

mindisp = df['displacement'].min()
print("\n The minimum displacement is : ",mindisp)

minwe = df['weight'].min()
print("\n The minimum weight is : ",minwe)

minacc = df['acceleration'].min()
print("\n The minimum acceleration is : ",minacc)

maxmpg = df['mpg'].max()
print("\n The maximum miles per gallon is : ",maxmpg)

maxcyl = df['cylinders'].max()
print("\n The maximum number of cylinders is : ",maxcyl)

maxdisp = df['displacement'].max()
print("\n The maximum displacement is : ",maxdisp)

maxwe = df['weight'].max()
print("\n The maximum weight is : ",maxwe)

maxacc = df['acceleration'].max()
print("\n The maximum acceleration is : ",maxacc)


