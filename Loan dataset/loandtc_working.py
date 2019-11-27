import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('loan_data.csv ')
df.info()

df.describe()
df.head()

plt.figure(figsize=(10,6))
df[df['credit.policy']==1]['fico'].hist(bins = 35 , color = 'blue',label = 'credit.policy 1',alpha = 0.6)
df[df['credit.policy']==0]['fico'].hist(bins = 35 , color = 'red',label = 'credit.policy 0',alpha =0.6)
plt.legend()
plt.xlabel('FICO')

plt.figure(figsize=(10,6))
df[df['not.fully.paid']==1]['fico'].hist(bins = 35 , color = 'blue',label = 'not.fully.paid 1',alpha = 0.6)
df[df['not.fully.paid']==0]['fico'].hist(bins = 35 , color = 'red',label = 'not.fully.paid 0',alpha =0.6)
plt.legend()
plt.xlabel('FICO')

cat_feats = ['purpose']
final_data = pd.get_dummies(df,columns = cat_feats,drop_first = True)

final_data.head(2)

X = final_data.drop('not.fully.paid',axis = 1)
y = final_data['not.fully.paid']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)
dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predicti = dtree.predict(X_test) 

print(classification_report(y_test,predicti))
print(confusion_matrix(y_test,predicti))

lm = LogisticRegression()
lm.fit(X_train,y_train)
lm_pred = lm.predict(X_test)
print("Logistic Regression")
print(classification_report(y_test,lm_pred))
print(confusion_matrix(y_test,lm_pred))

