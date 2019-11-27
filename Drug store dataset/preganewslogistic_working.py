import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix

preg = pd.read_csv('Data_Regression_Classification.csv')

preg.info()
preg.isnull().sum()
preg.columns
preg1 = preg.drop(['Loyalty_Card','Residence_Type','PREGNANT'],axis = 1)
preg1 = pd.get_dummies(preg1,columns = [ 'Implied_Gender','Celebrated_Valentines', 
                  'ThanksGiving_Shopping','Discount_Prg', 'Pregnancy_Test', 
                  'Number_Shopping_trips', 'Maternity_Clothes',
              'Birth_Control', 'Feminine_Hygiene',
       'Folic_Acid', 'Prenatal_Vitamins',
       'Prenatal_Yoga', 'Body_Pillow', 'Ginger_Ale', 'Nausea_tablets',
       'Stopped_buying_ciggies', 'Cigarettes', 'Smoking_Cessation',
       'Stopped_buying_wine', 'Wine'],drop_first = True)

X = StandardScaler().fit_transform(preg1)
pca = PCA(n_components=75)
pc = pca.fit_transform(X)

preg.corr()
preg.columns

preg.head()
X = pc
#X = preg.drop(['PREGNANT_Yes'],axis = 1)
y = preg['PREGNANT']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 101)

lm = LogisticRegression()
lm.fit(X_train,y_train)
predi = lm.predict(X_test)
accu = lm.score(X_test,y_test)

print(classification_report(y_test,predi))
print(confusion_matrix(y_test,predi))



