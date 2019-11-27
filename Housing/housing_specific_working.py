import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import math

df = pd.read_csv('USA_Housing.csv')
df.head()
df.info()
df.describe()
df.columns
X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y = df['Price']



X_train = X.loc[0:3000]
y_train = y.loc[0:3000]
X_test = X.loc[3001:]
y_test = y.loc[3001:]

lm = LinearRegression()
lm.fit(X_train,y_train)

predi = lm.predict(X_test)
predi
#plt.scatter(y_test,predi)


accuracy = lm.score(X_test,y_test)

print(accuracy*100) 

print(metrics.mean_absolute_error(y_test,predi))
print(math.sqrt(metrics.mean_squared_error(y_test,predi)))

cdf = pd.DataFrame(lm.coef_,X.columns,columns = ['coeff'])
print(cdf)

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

# solution
a, b = best_fit(y_test, predi)
#best fit line:
#y = 0.80 + 0.92x

#plot points and fit line
plt.scatter(y_test, predi)
yfit = [a + b * xi for xi in y_test]
plt.plot(y_test, yfit)