#Importing the required modules for the program
from tkinter import *
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from PIL import ImageTk, Image
from PIL import Image, ImageTk
#Creating the window
root = Tk()

#Defining the window size
root.geometry("8000x8000")

#Defining the window title
root.title("Stock Predictor By ticker")


#Globally defining the variables that are used in the functions and are of doube type
Open = DoubleVar()
High = DoubleVar()
Low = DoubleVar()
Close = DoubleVar()
Volume = DoubleVar()
Adjusted_Open = DoubleVar()
Adjusted_High = DoubleVar()
Adjusted_Low = DoubleVar()
Adjusted_Volume = DoubleVar()

#Function to display the predicted value of ticker 'A'
def a(Open,High,Low,Close,Volume,Adjusted_Open,Adjusted_High,Adjusted_Low,Adjusted_Volume):
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
      
     df = df.loc[df['ticker'] == 'A']
     
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
     labelp.grid(row = 4 , column = 4 , padx = 10 , pady = 10 )
      
     labela = Label(root,text = "The accuracy of this model is : "+str(accuracy*100))
     labela.grid(row = 5 , column = 4 , padx = 10 , pady = 10)     

#Function to take input values from the user
def a1():
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
    label1.grid(row = 1 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry1 = Entry(root, textvariable = Open)
    entry1.grid(row = 1 , column = 3 , padx = 10 , pady = 20)

    label2 = Label(root,text = "Enter the High value")
    label2.grid(row = 2 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry2 = Entry(root, textvariable = High)
    entry2.grid(row = 2 , column = 3 , padx = 10 , pady = 20)

    label3 = Label(root,text = "Enter the Low value")
    label3.grid(row = 3 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry3 = Entry(root, textvariable = Low)
    entry3.grid(row = 3 , column = 3 , padx = 10 , pady = 20)

    label4 = Label(root,text = "Enter the Close value")
    label4.grid(row = 4 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry4 = Entry(root,textvariable = Close)
    entry4.grid(row = 4 , column = 3 , padx = 10 , pady = 20)

    label5 = Label(root,text = "Enter the Volume value")
    label5.grid(row = 5 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry5 = Entry(root, textvariable = Volume)
    entry5.grid(row = 5 , column = 3  , padx = 10 , pady = 20)

    label6 = Label(root,text = "Enter the Adjusted Open value")
    label6.grid(row = 6 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry6 = Entry(root,textvariable = Adjusted_Open)
    entry6.grid(row = 6 , column = 3 , padx = 10 , pady = 20)

    label7 = Label(root,text = "Enter the Adjusted High value")
    label7.grid(row = 7 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry7 = Entry(root,textvariable = Adjusted_High)
    entry7.grid(row = 7 , column = 3 , padx = 10 , pady = 20)

    label8 = Label(root,text = "Enter the Adjusted Low value")
    label8.grid(row = 8 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry8 = Entry(root,textvariable = Adjusted_Low)
    entry8.grid(row = 8 , column = 3 , padx = 10 , pady = 20)

    label9 = Label(root,text = "Enter the Adjusted Volume value")
    label9.grid(row = 9 , column = 2 ,sticky = W , padx = 100 , pady = 20)
        
    entry9 = Entry(root,textvariable = Adjusted_Volume)
    entry9.grid(row = 9 , column = 3 , padx = 10 , pady = 20)

    button1 = Button(root , text = "Click to predict the ADJUSTED CLOSE VALUE",command = lambda : a(Open,High,Low,Close,Volume,Adjusted_Open,Adjusted_High,Adjusted_Low,Adjusted_Volume))
    button1.grid(row = 3 , column = 4 , pady = 10)

#Function to display the predicted value of ticker 'AA' 
def aa(Open,High,Low,Close,Volume,Adjusted_Open,Adjusted_High,Adjusted_Low,Adjusted_Volume):
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
      
     df = df.loc[df['ticker'] == 'AA']
     
     X1 = df[df['Date1'] > 2017]
     X2 = df[df['Date1'] <= 2017]
          
     X1 = X1[['open','high','low','close','volume','adj_open','adj_high','adj_low','adj_volume']]
     X2 = X2[['open','high','low','close','volume','adj_open','adj_high','adj_low','adj_volume']]
      
     y1 = df[df['Date1'] > 2017]
     y2 = df[df['Date1'] <= 2017]
         
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
     labelp.grid(row = 4 , column = 4 , padx = 10 , pady = 10 )
      
     labela = Label(root,text = "The accuracy of this model is : "+str(accuracy*100))
     labela.grid(row = 5 , column = 4 , padx = 10 , pady = 10)
    
#Function to take input values from the user    
def aa1():
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
    label1.grid(row = 1 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry1 = Entry(root, textvariable = Open)
    entry1.grid(row = 1 , column = 3 , padx = 10 , pady = 20)

    label2 = Label(root,text = "Enter the High value")
    label2.grid(row = 2 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry2 = Entry(root, textvariable = High)
    entry2.grid(row = 2 , column = 3 , padx = 10 , pady = 20)

    label3 = Label(root,text = "Enter the Low value")
    label3.grid(row = 3 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry3 = Entry(root, textvariable = Low)
    entry3.grid(row = 3 , column = 3 , padx = 10 , pady = 20)

    label4 = Label(root,text = "Enter the Close value")
    label4.grid(row = 4 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry4 = Entry(root,textvariable = Close)
    entry4.grid(row = 4 , column = 3 , padx = 10 , pady = 20)

    label5 = Label(root,text = "Enter the Volume value")
    label5.grid(row = 5 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry5 = Entry(root, textvariable = Volume)
    entry5.grid(row = 5 , column = 3  , padx = 10 , pady = 20)

    label6 = Label(root,text = "Enter the Adjusted Open value")
    label6.grid(row = 6 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry6 = Entry(root,textvariable = Adjusted_Open)
    entry6.grid(row = 6 , column = 3 , padx = 10 , pady = 20)

    label7 = Label(root,text = "Enter the Adjusted High value")
    label7.grid(row = 7 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry7 = Entry(root,textvariable = Adjusted_High)
    entry7.grid(row = 7 , column = 3 , padx = 10 , pady = 20)

    label8 = Label(root,text = "Enter the Adjusted Low value")
    label8.grid(row = 8 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry8 = Entry(root,textvariable = Adjusted_Low)
    entry8.grid(row = 8 , column = 3 , padx = 10 , pady = 20)

    label9 = Label(root,text = "Enter the Adjusted Volume value")
    label9.grid(row = 9 , column = 2 ,sticky = W , padx = 100 , pady = 20)
        
    entry9 = Entry(root,textvariable = Adjusted_Volume)
    entry9.grid(row = 9 , column = 3 , padx = 10 , pady = 20)

    button1 = Button(root , text = "Click to predict the ADJUSTED CLOSE VALUE",command = lambda:aa(Open,High,Low,Close,Volume,Adjusted_Open,Adjusted_High,Adjusted_Low,Adjusted_Volume))
    button1.grid(row = 3 , column = 4 , pady = 10) 

#Function to display the predicted value of ticker 'AAL'
def aal(Open,High,Low,Close,Volume,Adjusted_Open,Adjusted_High,Adjusted_Low,Adjusted_Volume):
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
      
     df = df.loc[df['ticker'] == 'AAL']
     
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
     labelp.grid(row = 4 , column = 4 , padx = 10 , pady = 10 )
      
     labela = Label(root,text = "The accuracy of this model is : "+str(accuracy*100))
     labela.grid(row = 5 , column = 4 , padx = 10 , pady = 10)    
 
#Function to take input values from the user    
def aal1():
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
    label1.grid(row = 1 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry1 = Entry(root, textvariable = Open)
    entry1.grid(row = 1 , column = 3 , padx = 10 , pady = 20)

    label2 = Label(root,text = "Enter the High value")
    label2.grid(row = 2 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry2 = Entry(root, textvariable = High)
    entry2.grid(row = 2 , column = 3 , padx = 10 , pady = 20)

    label3 = Label(root,text = "Enter the Low value")
    label3.grid(row = 3 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry3 = Entry(root, textvariable = Low)
    entry3.grid(row = 3 , column = 3 , padx = 10 , pady = 20)

    label4 = Label(root,text = "Enter the Close value")
    label4.grid(row = 4 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry4 = Entry(root,textvariable = Close)
    entry4.grid(row = 4 , column = 3 , padx = 10 , pady = 20)

    label5 = Label(root,text = "Enter the Volume value")
    label5.grid(row = 5 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry5 = Entry(root, textvariable = Volume)
    entry5.grid(row = 5 , column = 3  , padx = 10 , pady = 20)

    label6 = Label(root,text = "Enter the Adjusted Open value")
    label6.grid(row = 6 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry6 = Entry(root,textvariable = Adjusted_Open)
    entry6.grid(row = 6 , column = 3 , padx = 10 , pady = 20)

    label7 = Label(root,text = "Enter the Adjusted High value")
    label7.grid(row = 7 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry7 = Entry(root,textvariable = Adjusted_High)
    entry7.grid(row = 7 , column = 3 , padx = 10 , pady = 20)

    label8 = Label(root,text = "Enter the Adjusted Low value")
    label8.grid(row = 8 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry8 = Entry(root,textvariable = Adjusted_Low)
    entry8.grid(row = 8 , column = 3 , padx = 10 , pady = 20)

    label9 = Label(root,text = "Enter the Adjusted Volume value")
    label9.grid(row = 9 , column = 2 ,sticky = W , padx = 100 , pady = 20)
        
    entry9 = Entry(root,textvariable = Adjusted_Volume)
    entry9.grid(row = 9 , column = 3 , padx = 10 , pady = 20)

    button1 = Button(root , text = "Click to predict the ADJUSTED CLOSE VALUE",command = lambda : aal(Open,High,Low,Close,Volume,Adjusted_Open,Adjusted_High,Adjusted_Low,Adjusted_Volume))
    button1.grid(row = 3 , column = 4 , pady = 10)    

#Function to display the predicted value of ticker 'AAMC'
def aamc(Open,High,Low,Close,Volume,Adjusted_Open,Adjusted_High,Adjusted_Low,Adjusted_Volume):
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
      
     df = df.loc[df['ticker'] == 'AAMC']
     
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
     labelp.grid(row = 4 , column = 4 , padx = 10 , pady = 10 )
      
     labela = Label(root,text = "The accuracy of this model is : "+str(accuracy*100))
     labela.grid(row = 5 , column = 4 , padx = 10 , pady = 10)      

#Function to take input values from the user       
def aamc1():
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
    label1.grid(row = 1 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry1 = Entry(root, textvariable = Open)
    entry1.grid(row = 1 , column = 3 , padx = 10 , pady = 20)

    label2 = Label(root,text = "Enter the High value")
    label2.grid(row = 2 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry2 = Entry(root, textvariable = High)
    entry2.grid(row = 2 , column = 3 , padx = 10 , pady = 20)

    label3 = Label(root,text = "Enter the Low value")
    label3.grid(row = 3 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry3 = Entry(root, textvariable = Low)
    entry3.grid(row = 3 , column = 3 , padx = 10 , pady = 20)

    label4 = Label(root,text = "Enter the Close value")
    label4.grid(row = 4 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry4 = Entry(root,textvariable = Close)
    entry4.grid(row = 4 , column = 3 , padx = 10 , pady = 20)

    label5 = Label(root,text = "Enter the Volume value")
    label5.grid(row = 5 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry5 = Entry(root, textvariable = Volume)
    entry5.grid(row = 5 , column = 3  , padx = 10 , pady = 20)

    label6 = Label(root,text = "Enter the Adjusted Open value")
    label6.grid(row = 6 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry6 = Entry(root,textvariable = Adjusted_Open)
    entry6.grid(row = 6 , column = 3 , padx = 10 , pady = 20)

    label7 = Label(root,text = "Enter the Adjusted High value")
    label7.grid(row = 7 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry7 = Entry(root,textvariable = Adjusted_High)
    entry7.grid(row = 7 , column = 3 , padx = 10 , pady = 20)

    label8 = Label(root,text = "Enter the Adjusted Low value")
    label8.grid(row = 8 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry8 = Entry(root,textvariable = Adjusted_Low)
    entry8.grid(row = 8 , column = 3 , padx = 10 , pady = 20)

    label9 = Label(root,text = "Enter the Adjusted Volume value")
    label9.grid(row = 9 , column = 2 ,sticky = W , padx = 100 , pady = 20)
        
    entry9 = Entry(root,textvariable = Adjusted_Volume)
    entry9.grid(row = 9 , column = 3 , padx = 10 , pady = 20)

    button1 = Button(root , text = "Click to predict the ADJUSTED CLOSE VALUE",command = lambda : aamc(Open,High,Low,Close,Volume,Adjusted_Open,Adjusted_High,Adjusted_Low,Adjusted_Volume))
    button1.grid(row = 3 , column = 4 , pady = 10)

#Function to display the predicted value of ticker 'AAN'
def aan(Open,High,Low,Close,Volume,Adjusted_Open,Adjusted_High,Adjusted_Low,Adjusted_Volume):
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
      
     df = df.loc[df['ticker'] == 'AAN']
     
     X1 = df[df['Date1'] > 1986]
     X2 = df[df['Date1'] <= 1986]
          
     X1 = X1[['open','high','low','close','volume','adj_open','adj_high','adj_low','adj_volume']]
     X2 = X2[['open','high','low','close','volume','adj_open','adj_high','adj_low','adj_volume']]
      
     y1 = df[df['Date1'] > 1986]
     y2 = df[df['Date1'] <= 1986]
         
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
     labelp.grid(row = 4 , column = 4 , padx = 10 , pady = 10 )
      
     labela = Label(root,text = "The accuracy of this model is : "+str(accuracy*100))
     labela.grid(row = 5 , column = 4 , padx = 10 , pady = 10)          

#Function to take input values from the user
def aan1():
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
    label1.grid(row = 2 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry1 = Entry(root, textvariable = Open)
    entry1.grid(row = 2 , column = 3 , padx = 10 , pady = 20)

    label2 = Label(root,text = "Enter the High value")
    label2.grid(row = 3 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry2 = Entry(root, textvariable = High)
    entry2.grid(row = 3 , column = 3 , padx = 10 , pady = 20)

    label3 = Label(root,text = "Enter the Low value")
    label3.grid(row = 4 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry3 = Entry(root, textvariable = Low)
    entry3.grid(row = 4 , column = 3 , padx = 10 , pady = 20)

    label4 = Label(root,text = "Enter the Close value")
    label4.grid(row = 5 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry4 = Entry(root,textvariable = Close)
    entry4.grid(row = 5 , column = 3 , padx = 10 , pady = 20)

    label5 = Label(root,text = "Enter the Volume value")
    label5.grid(row = 6 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry5 = Entry(root, textvariable = Volume)
    entry5.grid(row = 6 , column = 3  , padx = 10 , pady = 20)

    label6 = Label(root,text = "Enter the Adjusted Open value")
    label6.grid(row = 7 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry6 = Entry(root,textvariable = Adjusted_Open)
    entry6.grid(row = 7 , column = 3 , padx = 10 , pady = 20)

    label7 = Label(root,text = "Enter the Adjusted High value")
    label7.grid(row = 8 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry7 = Entry(root,textvariable = Adjusted_High)
    entry7.grid(row = 8 , column = 3 , padx = 10 , pady = 20)

    label8 = Label(root,text = "Enter the Adjusted Low value")
    label8.grid(row = 9 , column = 2 , sticky = W , padx = 100 , pady = 20)

    entry8 = Entry(root,textvariable = Adjusted_Low)
    entry8.grid(row = 9 , column = 3 , padx = 10 , pady = 20)

    label9 = Label(root,text = "Enter the Adjusted Volume value")
    label9.grid(row = 10 , column = 2 ,sticky = W , padx = 100 , pady = 20)
        
    entry9 = Entry(root,textvariable = Adjusted_Volume)
    entry9.grid(row = 10 , column = 3 , padx = 10 , pady = 20)

    button1 = Button(root , text = "Click to predict the ADJUSTED CLOSE VALUE",command = lambda : aan(Open,High,Low,Close,Volume,Adjusted_Open,Adjusted_High,Adjusted_Low,Adjusted_Volume))
    button1.grid(row = 3 , column = 4 , pady = 10)

#Initaling a varibale to integer type which is uded in the radiobuttons
var = IntVar()

 #To show what the uer has to do in the begining   
label1 = Label(root,text = "Select the ticker which you want to and predict")
label1.grid(row = 0 , column = 3 , sticky = W , padx = 200 , pady = 30)

#The radio button options that the ui provides the user to select
r1 = Radiobutton(root,text = "A",variable = var , value = 1, command = a1)
r1.grid(row = 1,column = 0 , sticky = W , padx = 20 , pady = 30)

r2 = Radiobutton(root,text = "AA",variable = var , value = 2, command = aa1)
r2.grid(row = 2,column = 0 , sticky = W , padx = 20 , pady = 30)

r3 = Radiobutton(root,text = "AAL",variable = var , value = 3, command = aal1)
r3.grid(row = 3,column = 0 , sticky = W , padx = 20 , pady = 30)

r4 = Radiobutton(root,text = "AAMC",variable = var , value = 4, command = aamc1)
r4.grid(row = 4,column = 0 , sticky = W , padx = 20 , pady = 30)

r5 = Radiobutton(root,text = "AAN",variable = var , value = 5, command = aan1)
r5.grid(row = 5,column = 0 , sticky = W , padx = 20 , pady = 30)

image = Image.open("daimler.jpg")
photo = ImageTk.PhotoImage(image)
label = Label(root,image=photo)
label.image = photo 
label.grid(row = 1 , column = 4)
#To run the ui program
root.mainloop()