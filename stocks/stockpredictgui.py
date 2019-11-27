from tkinter import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

root = Tk()
root.geometry("1000x1000")
root.title("Stocks")

Open = DoubleVar()
High = DoubleVar()
Low = DoubleVar()
Close = DoubleVar()
Volume = DoubleVar()
Adjusted_Open = DoubleVar()
Adjusted_High = DoubleVar()
Adjusted_Low = DoubleVar()
Adjusted_Volume = DoubleVar()

label1 = Label(root,text = "Enter the Open value")
label1.grid(row = 0 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry1 = Entry(root, textvariable = Open)
entry1.grid(row = 0 , column = 2 , padx = 10 , pady = 10)



label2 = Label(root,text = "Enter the High value")
label2.grid(row = 1 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry2 = Entry(root, textvariable = High)
entry2.grid(row = 1 , column = 2 , padx = 10 , pady = 10)

label3 = Label(root,text = "Enter the Low value")
label3.grid(row = 2 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry3 = Entry(root, textvariable = Low)
entry3.grid(row = 2 , column = 2 , padx = 10 , pady = 10)

label4 = Label(root,text = "Enter the Close value")
label4.grid(row = 3 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry4 = Entry(root,textvariable = Close)
entry4.grid(row = 3 , column = 2 , padx = 10 , pady = 10)

label5 = Label(root,text = "Enter the Volume value")
label5.grid(row = 4 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry5 = Entry(root, textvariable = Volume)
entry5.grid(row = 4 , column = 2  , padx = 10 , pady = 10)

label6 = Label(root,text = "Enter the Adjusted Open value")
label6.grid(row = 5 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry6 = Entry(root,textvariable = Adjusted_Open)
entry6.grid(row = 5 , column = 2 , padx = 10 , pady = 10)

label7 = Label(root,text = "Enter the Adjusted High value")
label7.grid(row = 6 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry7 = Entry(root,textvariable = Adjusted_High)
entry7.grid(row = 6 , column = 2 , padx = 10 , pady = 10)

label8 = Label(root,text = "Enter the Adjusted Low value")
label8.grid(row = 7 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry8 = Entry(root,textvariable = Adjusted_Low)
entry8.grid(row = 7 , column = 2 , padx = 10 , pady = 10)

label9 = Label(root,text = "Enter the Adjusted Volume value")
label9.grid(row = 8 , column = 0 ,sticky = W , padx = 10 , pady = 10)

entry9 = Entry(root,textvariable = Adjusted_Volume)
entry9.grid(row = 8 , column = 2 , padx = 10 , pady = 10)





def spt(Open,High,Low,Close,Volume,Adjusted_Open,Adjusted_High,Adjusted_Low,Adjusted_Volume):
    
    Open1 = Open.get()
    High1 = High.get()
    Low1 = Low.get()
    Close1 = Close.get()
    Volume1 = Volume.get()
    Adjusted_Open1 = Adjusted_Open.get()
    Adjusted_High1 = Adjusted_High.get()
    Adjusted_Low1 = Adjusted_Low.get()
    Adjusted_Volume1 = Adjusted_Volume.get()
    
    value_list = [[Open1,High1,Low1,Close1,Volume1,Adjusted_Open1,Adjusted_High1,Adjusted_Low1,Adjusted_Volume1]]
    df = pd.read_csv('STOCK-PRICES.csv')
               
                         
    date = df['date']
                
    date1 = []
    for i in date:
        a = i.split('/')
        b = int(a[2])
        date1.append(b)
                    
    df['Date1'] = pd.DataFrame(date1)


    X1 = df[df['Date1'] > 2015]
    X2 = df[df['Date1'] <= 2015]


    X1 = X1[['open','high','low','close','volume','adj_open','adj_high','adj_low','adj_volume']]
    X2 = X2[['open','high','low','close','volume','adj_open','adj_high','adj_low','adj_volume']]

    y1 = df[df['Date1'] > 2015]
    y2 = df[df['Date1'] <= 2015]
    
    y1 = y1[['adj_close']]
    y2 = y2[['adj_close']]

    X_train = X2
    X_test = X1

    y_train = y2
    y_test = y1

    lm = LinearRegression()
    
    lm.fit(X_train,y_train)
    
    arrvalue = np.array(value_list)

    dff1 = pd.DataFrame(arrvalue)

    prediction1 = float(lm.predict(dff1))
    accuracy = lm.score(X_test,y_test)
    
    labelp = Label(root,text = "The predicted ADJUSTED CLOSE value is : "+str(prediction1))
    labelp.grid(row = 12 , column = 3 , padx = 10 , pady = 10 )
    
    labela = Label(root,text = "The accuracy of this model is : "+str(accuracy*100))
    labela.grid(row = 14 , column = 3 , padx = 10 , pady = 10)
    
    
button1 = Button(root,text = "Click to find the Adjusted Close Value ",command = lambda : spt(Open,High,Low,Close,Volume,Adjusted_Open,Adjusted_High,Adjusted_Low,Adjusted_Volume))
button1.grid(row = 10 , column = 3,padx = 10, pady = 10)



root.mainloop()
