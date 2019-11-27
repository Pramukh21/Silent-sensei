#Importing the required packages
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
#Read the csv file
retail_set = pd.read_csv('RetailDataSet.csv') 

#Check out the columns 
retail_set.columns
#Droping out the coulmn 'year' as it is not required
retail_set = retail_set.drop('YEAR',axis = 1)

#filling out the missing values
retail_set.fillna(retail_set.mean(),inplace = True)

#converting the categorical values to numeric values so as to include in the model
X = pd.get_dummies(retail_set,columns = ['CITY','STATE','FORMAT','REGION','SPECIAL'],drop_first = True)

X = StandardScaler().fit_transform(X)

#Reducing the dimensions by taking the principal components
pca = PCA(n_components= 20)
pc = pca.fit_transform(X)

#Assigning the required components to features and output labels
X = pc
y = retail_set['RS_SALES']

#Divviding the file into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 101)

#Defining the linear regression model
lm = LinearRegression()

#Training the model
lm.fit(X_train,y_train)

#Predicting the RSVALUES using the test values
predi = lm.predict(X_test)

#Calculating the accuracy of the model
acc = lm.score(X_test,y_test)

rms = sqrt(mean_squared_error(y_test,predi))
r2 = r2_score(y_test,predi)
print(r2)
print(rms)

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