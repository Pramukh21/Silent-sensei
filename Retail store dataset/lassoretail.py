import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
from sklearn.preprocessing import StandardScaler 

retail_set = pd.read_csv('RetailDataSet.csv')
reatil_set= retail_set.fillna(retail_set.mean(),inplace = True)
retail_set = retail_set.drop(['YEAR'],axis = 1)
retail_set.isnull().any()
retail_set.corr()
#X1 = retail_set.drop(['RS_SALES','CITY', 'STATE', 'FORMAT',
      # 'REGION', 'SPECIAL'],axis = 1)
      
X = pd.get_dummies(retail_set,columns = ['CITY','STATE','FORMAT','REGION','SPECIAL'],drop_first = True)

X1 = StandardScaler().fit_transform(X)
y = retail_set['RS_SALES']


X_train,X_test,y_train,y_test = train_test_split(X1,y,test_size = 0.2,random_state = 101)

lm = Lasso(alpha = 1)
a = lm.fit(X_train,y_train)
predi = lm.predict(X_test)

accu = lm.score(X_test,y_test)

rms = sqrt(mean_squared_error(y_test,predi))
r2 = r2_score(y_test,predi)

#0.9848830686632989