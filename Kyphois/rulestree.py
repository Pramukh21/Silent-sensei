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

from sklearn import tree
tree.export_graphviz(dtree,out_file = 'treevis.dot')
