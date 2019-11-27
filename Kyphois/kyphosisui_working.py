import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from tkinter import *

root = Tk()
root.geometry("800x800")
root.title('Kyphosis Predictor')

kyphosis = pd.read_csv('kyphosis.csv')

kyphosis.info()

kyphosis.columns

age = IntVar()
number = IntVar()
start = IntVar()

def kypho(age,number,start):
    age1 = age.get()
    number1 = number.get()
    start1 = start.get()
    
    value_list = [[age1,number1,start1]]
    
    X = kyphosis.drop(['Kyphosis'],axis = 1)
    y = kyphosis['Kyphosis']
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train,y_train)
    predi = dtree.predict(value_list)
    accuracy = dtree.score(X_test,y_test)
    predi = "".join(predi)
    
    labelp = Label(root,text = "Kyphosis is "+str(predi))
    labelp.grid(row = 6,column = 2)

label1 = Label(root,text = "Enter the age of the paitent ")
label1.grid(row = 1, column = 1,padx = 10, pady = 10,sticky = W)

entry1 = Entry(root,textvariable = age)
entry1.grid(row = 1,column = 2,padx = 10 ,pady = 10)

label2 = Label(root,text = "Enter the Number ")
label2.grid(row = 2,column = 1,padx = 10,pady = 10,sticky = W)

entry2 = Entry(root,textvariable = number)
entry2.grid(row = 2,column = 2 ,padx = 10 ,pady = 10)

label3 = Label(root,text = "Enter the Start number ")
label3.grid(row = 3 ,column = 1,padx = 10 ,pady = 10 , sticky = W)

entry3 = Entry(root,textvariable = start)
entry3.grid(row = 3 ,column = 2,padx=10 ,pady =10)

button1 = Button(root,text = "Enter to predict wether kyphosis is present or not",command = lambda :kypho(age,number,start))

button1.grid(row = 4,column = 2 ,padx = 10 ,pady = 10)

root.mainloop()

