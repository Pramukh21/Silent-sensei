import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('titanic_train.csv')
train.head()

train.info()

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else :
            return 24
    else:
        return Age
    
    
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis = 1)
print(train['Age'])
train['Age'].isnull()
train.drop('Cabin',axis = 1 ,inplace = True)
train.dropna(inplace = True)
sex = pd.get_dummies(train['Sex'],drop_first = True)
embark = pd.get_dummies(train['Embarked'],drop_first = True)
embark.head()

train = pd.concat([train,sex,embark],axis = 1)
#print(train.head())

train.drop(['Sex','Embarked','Name','Ticket'],axis = 1,inplace = True)
print(train.head())

train.drop(['PassengerId'],axis = 1,inplace = True)
print(train.head())

X = train.drop('Survived',axis = 1)
y = train['Survived']
X.columns

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3 ,random_state =101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

prediction = logmodel.predict(X_test)

accuracy = logmodel.score(X_test,y_test)
print(accuracy)

from sklearn.metrics import classification_report

print(classification_report(y_test,prediction))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,prediction))


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)
lr_pred = lm.predict(X_test)
lr_sc = lm.score(X_test,y_test)

print("Linear Regression",lr_sc*100)
print(metrics.mean_absolute_error(y_test,lr_pred))
print(metrics.mean_squared_error(y_test,lr_pred))

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
a, b = best_fit(y_test, lr_pred)
#best fit line:
#y = 0.80 + 0.92x

#plot points and fit line
plt.scatter(y_test, lr_pred)
yfit = [a + b * xi for xi in y_test]
plt.plot(y_test, yfit)


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dtr_pred = dtree.predict(X_test)
print("Decision Tree")
print(classification_report(y_test,dtr_pred))

print(confusion_matrix(y_test,dtr_pred))


from sklearn.neighbors import KNeighborsClassifier

error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))
    
    

plt.figure(figsize = (10,6))
plt.plot(range(1,40),error_rate,color = 'blue',marker ='o',ls = '--',markersize = 10,markerfacecolor = 'red')
plt.title('ERROR RATE VS K VALUE')
plt.xlabel('K')
plt.ylabel('Eroor rate')
print("KNN")
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train,y_train)
knnpred = knn.predict(X_test)
print(confusion_matrix(y_test,knnpred))
print(classification_report(y_test,knnpred))