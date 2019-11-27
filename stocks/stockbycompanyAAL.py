import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('STOCK-PRICES.csv')
dfaal = df.loc[df['ticker'] == 'AAL']
print(dfaal.head())

date = dfaal['date']

date1 = []
for i in date:
    a = i.split('/')
    b = int(a[2])
    date1.append(b)

dfaal['Date1'] = pd.DataFrame(date1)
print(dfaal.head())

X1 = dfaal[dfaal['Date1'] > 2015]
X2 = dfaal[dfaal['Date1'] <= 2015]


X1 = X1[['open','high','low','close','volume','adj_open','adj_high','adj_low','adj_volume']]
X2 = X2[['open','high','low','close','volume','adj_open','adj_high','adj_low','adj_volume']]

y1 = dfaal[dfaal['Date1'] > 2015]
y2 = dfaal[dfaal['Date1'] <= 2015]

y1 = y1[['adj_close']]
y2 = y2[['adj_close']]

X_train = X2
X_test = X1

y_train = y2
y_test = y1

lm = LinearRegression()

lm.fit(X_train,y_train)
Open = input("\n Enter the Open value : ")
High = input("\n Enter the High value : ")
Low = input("\n Enter the Low value : ")
Close = input("\n Enter the Close value : ")
Volume = input("\n Enter the Volume value : ")
Adj_Open = input("\n Enter the Adjusted Open value : ")
Adj_High = input("\n Enter the Adjusted High value : ")
Adj_Low = input("\n Enter the Adjusted Low value : ")
Adj_Volume = input("\n Enter the Adjusted Volume value : ")

value_list = [[Open,High,Low,Close,Volume,Adj_Open,Adj_High,Adj_Low,Adj_Volume]]

arrvalue = np.array(value_list)

dff1 = pd.DataFrame(arrvalue)

prediction1 = lm.predict(dff1)
print("\nThe predicted ADJ_CLOSE  value is : ",float(prediction1))
accuracy = lm.score(X_test,y_test)

print("\nThe accuracy obtained for my model is  : ",accuracy*100)