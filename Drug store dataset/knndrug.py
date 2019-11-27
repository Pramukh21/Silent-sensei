import numpy as np
import pandas  as pd 
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


preg = pd.read_csv('Data_Regression_Classification.csv')
preg = preg.drop(['Loyalty_Card','Residence_Type'],axis = 1)
preg.columns

preg = pd.get_dummies(preg,columns = [ 'Implied_Gender','Celebrated_Valentines', 
                  'ThanksGiving_Shopping','Discount_Prg', 'Pregnancy_Test', 
                  'Number_Shopping_trips', 'Maternity_Clothes',
              'Birth_Control', 'Feminine_Hygiene',
       'Folic_Acid', 'Prenatal_Vitamins',
       'Prenatal_Yoga', 'Body_Pillow', 'Ginger_Ale', 'Nausea_tablets',
       'Stopped_buying_ciggies', 'Cigarettes', 'Smoking_Cessation',
       'Stopped_buying_wine', 'Wine','PREGNANT'],drop_first = True)

X = StandardScaler().fit_transform(preg)
pca = PCA(n_components=30)
pc = pca.fit_transform(X)

X = pc
y = preg['PREGNANT_Yes']
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size = 0.2,random_state = 101)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

predi = dtree.predict(X_test)
accu = dtree.score(X_test,y_test)

print(classification_report(y_test,predi))

print("\n\n",confusion_matrix(y_test,predi))



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

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
predknn = knn.predict(X_test)
print("KNN\n")
print(classification_report(y_test,predi))

print("\n\n",confusion_matrix(y_test,predi))