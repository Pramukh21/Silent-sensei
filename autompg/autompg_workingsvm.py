import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('autompg2.csv')
print(df.head())
df.info()
print(df.describe())
df.columns

X = df[['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration','origin','modelyear']]


X=StandardScaler().fit_transform(X)
pca = PCA(n_components=7)

pc = pca.fit_transform(X)
X = pc
y = df['mpg']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)

clf = SVR(C=100,epsilon=0.1)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
accuracy = clf.score(X_test,y_test)
print(accuracy)

para_grid = {'C':[1.0,10,100,100],'epsilon':[0.1,0.01,0.001],'gamma':[0.1,0.01,0.001]}
grid = GridSearchCV(SVR(),para_grid,verbose = 3)
grid.fit(X_train,y_train)
grid_pred = grid.predict(X_test)
print(grid.score(X_test,y_test))