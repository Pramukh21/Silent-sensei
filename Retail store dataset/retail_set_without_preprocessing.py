import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from math import sqrt


retail_set = pd.read_csv('RetailDataSet.csv')
reatil_set= retail_set.fillna(retail_set.mean(),inplace = True)
retail_set = retail_set.drop(['YEAR'],axis = 1)
retail_set.isnull().any()
retail_set.corr()
X1 = retail_set.drop(['RS_SALES','CITY', 'STATE', 'FORMAT',
       'REGION', 'SPECIAL'],axis = 1)

y = retail_set['RS_SALES']


X_train,X_test,y_train,y_test = train_test_split(X1,y,test_size = 0.2,random_state = 101)
X_train.min()
lm = LinearRegression()
lm.fit(X_train,y_train)
predi = lm.predict(X_test)

acu = lm.score(X_test,y_test)

#plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, predi, 1))(np.unique(y_test)))
#plt.scatter(np.unique(y_test), np.poly1d(np.polyfit(y_test, predi, 1))(np.unique(y_test)))

# sample points 
#X = [0, 5, 10, 15, 20]
#Y = [0, 7, 10, 13, 20]

# solve for a and b
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

# plot points and fit line
plt.scatter(y_test, predi)
yfit = [a + b * xi for xi in y_test]
plt.plot(y_test, yfit)



print("Mean Absolute Eroor =",mean_absolute_error(y_test,predi))

print('Root Mean Squared Error =',np.sqrt(mean_squared_error(y_test,predi)))