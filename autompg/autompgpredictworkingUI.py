from tkinter import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split



root = Tk()
root.geometry("1000x1000")
root.title("MPG predictor")

cylinders = IntVar()
displacement = DoubleVar()
horsepower = DoubleVar()
weight = DoubleVar()
acceleration = DoubleVar()
modelyear = DoubleVar()
origin = IntVar()

label1 = Label(root,text = "Enter the Number of cylinders ")
label1.grid(row = 0 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry1 = Entry(root, textvariable = cylinders)
entry1.grid(row = 0 , column = 2 , padx = 10 , pady = 10)

label2 = Label(root,text = "Enter the displacement of the engine ")
label2.grid(row = 1 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry2 = Entry(root, textvariable = displacement)
entry2.grid(row = 1 , column = 2 , padx = 10 , pady = 10)

label3 = Label(root,text = "Enter the Horsepower")
label3.grid(row = 2 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry3 = Entry(root, textvariable = horsepower)
entry3.grid(row = 2 , column = 2 , padx = 10 , pady = 10)

label4 = Label(root,text = "Enter the Weight")
label4.grid(row = 3 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry4 = Entry(root,textvariable = weight)
entry4.grid(row = 3 , column = 2 , padx = 10 , pady = 10)

label5 = Label(root,text = "Enter the acceleration of the vehicle")
label5.grid(row = 4 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry5 = Entry(root, textvariable = acceleration)
entry5.grid(row = 4 , column = 2  , padx = 10 , pady = 10)

label6 = Label(root,text = "Enter Car Model Year")
label6.grid(row = 5 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry6 = Entry(root,textvariable = modelyear)
entry6.grid(row = 5 , column = 2 , padx = 10 , pady = 10)

label7 = Label(root,text = "Enter the origin 1:American 2:European 3:Asian")
label7.grid(row = 6 , column = 0 , sticky = W , padx = 10 , pady = 10)

entry7 = Entry(root,textvariable = origin)
entry7.grid(row = 6 , column = 2 , padx = 10 , pady = 10)


def spt(cylinders,displacement,horsepower,weight,acceleration,modelyear,origin):
    cylinders1 = cylinders.get()
    displacement1 = displacement.get()
    horsepower1 = horsepower.get()
    weight1 = weight.get()
    acceleration1 = acceleration.get()
    modelyear1 = modelyear.get()
    origin1 = origin.get()
    
    df = pd.read_csv('autompg2.csv')
    
    value_list = [[cylinders1, displacement1, horsepower1, weight1,
       acceleration1,origin1,modelyear1]]
    arrval = np.array(value_list)
    
    X = df[['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration','origin','modelyear']]

    y = df['mpg']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)

    lm = LinearRegression()
    lm.fit(X_train,y_train)
    dff1 = pd.DataFrame(arrval)
    prediction = float(lm.predict(dff1))
    
    accuracy = lm.score(X_test,y_test)
   
    
    labelp = Label(root,text = "The predicted Mileage(Miles Per Gallon) is : "+str(prediction))
    labelp.grid(row = 12 , column = 3 , padx = 10 , pady = 10 )
    
    labela = Label(root,text = "The accuracy of this model is : "+str(accuracy*100))
    labela.grid(row = 14 , column = 3 , padx = 10 , pady = 10)

button1 = Button(root,text = "Click to find the the Mileage(miles per gallon) ",command = lambda : spt(cylinders,displacement,horsepower,weight,acceleration,modelyear,origin))
button1.grid(row = 10 , column = 3,padx = 10, pady = 10)
root.mainloop()
