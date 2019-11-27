import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import metrics

df = pd.read_csv('USA_Housing.csv')
df.head()
df.info()
df.describe()
df.columns
X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y = df['Price']

print(X.loc[0:10])
print(y.loc[0:10])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 101)

lm = LinearRegression()
lm.fit(X_train,y_train)
lm.intercept_
lm.coef_
X_train.columns

cdf = pd.DataFrame(lm.coef_,X.columns,columns = ['Coeff'])
cdf

predictions = lm.predict(X_test)
len(predictions)
plt.scatter(y_test,predictions)
y_test
predictions
accuracy = lm.score(X_test,y_test)

print(accuracy*100) 

print(metrics.mean_absolute_error(y_test,predictions))
print(metrics.mean_squared_error(y_test,predictions))