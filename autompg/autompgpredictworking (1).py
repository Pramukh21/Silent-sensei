import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv('autompg2.csv')
print(df.head())
df.info()
print(df.describe())
df.columns

X = df[['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration','origin','modelyear']]
X=StandardScaler().fit_transform(X)
pca = PCA(n_components=7)

pc = pca.fit_transform(X)
X = pc
y = df['mpg']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)

lm = LinearRegression()
lm.fit(X_train,y_train)

predi = lm.predict(X_test)

accuracy = lm.score(X_test,y_test)
print("\n The accuracy is : ",accuracy)
print("The coefficents are : ",lm.coef_)
print("The intercept is : ",lm.intercept_)

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

print("mae=",metrics.mean_absolute_error(y_test,predi))
print(metrics.mean_squared_error(y_test,predi))
print('rmse=',np.sqrt(metrics.mean_squared_error(y_test,predi)))




