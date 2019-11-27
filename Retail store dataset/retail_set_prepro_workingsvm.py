from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
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

#sel = VarianceThreshold(threshold=(.5* (1 - .5)))

#sel.fit(X)

#X = sel.transform(X)
X = StandardScaler().fit_transform(X)

#To reduce the dimensions using principal component analyis
pca = PCA(n_components= 3)
pc = pca.fit_transform(X)

#Assigning the required components to features and output labels so that it can be used in future for training and testing
X = pc
y = retail_set['RS_SALES']

#Dividing the set into training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)


clf = SVR()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
accuracy = clf.score(X_test,y_test)
print(accuracy)

para_grid = {'C':[1.0,10,100,100],'epsilon':[0.1,0.01,0.001],'gamma':[0.1,0.01,0.001]}
grid = GridSearchCV(SVR(),para_grid,verbose = 3)
grid.fit(X_train,y_train)
grid_pred = grid.predict(X_test)
print(grid.score(X_test,y_test))