import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight')

preg = pd.read_csv('Data_Regression_Classification.csv')

preg.info()
preg.isnull().sum()
preg.columns
preg = preg.drop(['Loyalty_Card','Residence_Type',],axis = 1)
preg = pd.get_dummies(preg,columns = [ 'Implied_Gender',
       'Discount_Prg', 'Celebrated_Valentines', 'Pregnancy_Test',
       'Birth_Control', 'Feminine_Hygiene', 'ThanksGiving_Shopping',
       'Number_Shopping_trips', 'Folic_Acid', 'Prenatal_Vitamins',
       'Prenatal_Yoga', 'Body_Pillow', 'Ginger_Ale', 'Nausea_tablets',
       'Stopped_buying_ciggies', 'Cigarettes', 'Smoking_Cessation',
       'Stopped_buying_wine', 'Wine', 'Maternity_Clothes', 'PREGNANT'],drop_first = True)

preg.corr()
preg.columns

preg.head()
X = preg.drop(['PREGNANT_Yes'],axis = 1)
y = preg['PREGNANT_Yes']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

lm = LinearRegression()
lm.fit(X_train,y_train)
predi = lm.predict(X_test)
accu = lm.score(X_test,y_test)

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

print("Mean Absolute Eroor =",mean_absolute_error(y_test,predi))

print('Root Mean Squared Error =',np.sqrt(mean_squared_error(y_test,predi)))

