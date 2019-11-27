#Importing the required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error,mean_squared_error

#Read the required CSV file
retail_set = pd.read_csv('RetailDataSet.csv') 

#To check what columns are present in the dataset
retail_set.columns

#To check the variance of each column
retail_set.var()

#To drop the year column
retail_set = retail_set.drop('YEAR',axis = 1)

#To fill the missing values with the mean of their respective columns
retail_set.fillna(retail_set.mean(),inplace = True)

#To convert the categorical values to numeric so that it can be included for the prediction
X = pd.get_dummies(retail_set,columns = ['CITY','STATE','FORMAT','REGION','SPECIAL'],drop_first = True)

#Eliminate the columns whose variance is less than 0.5
sel = VarianceThreshold(threshold=(.5* (1 - .5)))

sel.fit(X)

X = sel.transform(X)
X = StandardScaler().fit_transform(X)

#To reduce the dimensions using principal component analyis
pca = PCA(n_components= 9)
pc = pca.fit_transform(X)

#Assigning the required components to features and output labels so that it can be used in future for training and testing
X = pc
y = retail_set['RS_SALES']

#Dividing the set into training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 101)

#Defining the linear regression model
lm = LinearRegression()

#Training the model
lm.fit(X_train,y_train)

#Predicting the RS_SALES values for the testing values
predi = lm.predict(X_test)

#Compared to get the accuracy of the model
acc = lm.score(X_test,y_test)

rms = sqrt(mean_squared_error(y_test,predi))
print(rms)

r2 = r2_score(y_test,predi)
print(r2)
#plotting for best fit line


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

