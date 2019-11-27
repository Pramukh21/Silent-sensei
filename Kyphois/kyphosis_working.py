import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('kyphosis.csv')
df.head()
df.info()

from sklearn.cross_validation import train_test_split

X = df.drop('Kyphosis',axis = 1)
y = df['Kyphosis']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
sco = dtree.score(X_test,y_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

from sklearn.linear_model import LogisticRegression
print("Logistic Regression\n")
lm = LogisticRegression()
lm.fit(X_train,y_train)
lm_pre = lm.predict(X_test)
print(confusion_matrix(y_test,lm_pre))
print(classification_report(y_test,lm_pre))

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

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
knnpred = knn.predict(X_test)
print(confusion_matrix(y_test,knnpred))
print(classification_report(y_test,knnpred))