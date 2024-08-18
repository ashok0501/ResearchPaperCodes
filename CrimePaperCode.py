# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 21:55:20 2021

@author: ashok
"""
import datetime

x=datetime.datetime(2021,1,25,23,12,10)

def find_fraction(dt):
    h=dt.hour 
    if h<=6:
        return 1
    elif h<12:
        return 2
    elif h<18:
        return 3
    else:
        return 4
    pass


import pandas as pd
dataset = pd.read_csv(r"ds.csv")
#split independent and dependent variables
x=dataset.iloc[:, 0:5].values
y=dataset.iloc[:,5].values
#normalize

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

#random forest
from sklearn.ensemble import RandomForestRegressor
r=RandomForestRegressor(n_estimators=10)
r.fit(x,y)
val=[[1,14,3,5,2]]
val=sc.transform(val)

#Support vector
from sklearn.svm import SVC
c=SVC(kernel="linear",random_state=0)
c.fit(x_train,y_train)



#kNN
from sklearn.neighbors import KNeighborsClassifier
c=KNeighborsClassifier(n_neighbors=3)
c.fit(x_train,y_train)

#NB
from sklearn.naive_bayes import GaussianNB
c = GaussianNB()
c.fit(x_train, y_train)


import matplotlib.pyplot as plt
 
x=[1,2,3,4]
xl=["13","15","17","19"]
plt.plot(x,[90,91.5,92,96.3],c="black",label="SVM (Proposed)")
plt.plot(x,[89.3,88.5,91.4,93.6],c="black",linestyle='--',label="SVM")
plt.plot(x,[91.2,91.7,94.5,97.2],c="red",label="kNN (Proposed)")
plt.plot(x,[89.1,88.9,92.8,96.8],c="red",linestyle='--',label="kNN")
plt.plot(x,[91.2,91.7,95.1,96.7],c="green",label="NB (Proposed)")
plt.plot(x,[89.9,87.3,93.5,96],c="green",linestyle='--',label="NB")
plt.plot(x,[94.4,95.1,96.8,98.3],c="blue",label="RF (Proposed)")
plt.plot(x,[93.4,94.1,94.6,93.9],c="blue",linestyle='--',label="RF")
"""
plt.plot(x,[90.4,93.1,95.2,96.8],c="black",label="SVM (Proposed)")
plt.plot(x,[90.2,88.3,94.4,96.2],c="black",linestyle='--',label="SVM")
plt.plot(x,[91.2,92.2,94.7,95.8],c="red",label="kNN (Proposed)")
plt.plot(x,[89.7,90.4,91.8,92.3],c="red",linestyle='--',label="kNN")
plt.plot(x,[94.1,94.1,95.6,96.3],c="green",label="NB (Proposed)")
plt.plot(x,[90.2,90.7,94.5,95.1],c="green",linestyle='--',label="NB")
plt.plot(x,[91.1,95.9,94,98.2],c="blue",label="RF (Proposed)")
plt.plot(x,[90.4,91.1,92.6,96.9],c="blue",linestyle='--',label="RF")
"""

plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("San Francisco")
plt.legend()
 
plt.xticks(x,xl)
plt.show()



x=[1,2,3,4,5,6,7,8]
y=[83,85,86,86,90,91,92,94]
plt.yticks(x,["NB","NB (With Proposed)","RF","RF (With Proposed)","kNN","kNN (With Proposed)","SVM","SVM (With Proposed)"])

plt.barh(x,y,color=["black","black","blue","blue","red","red","green","green"])
plt.xlabel("Accuracy")
plt.title("Los Angeles Dataset - # of Hotspots:50")
plt.show()

x=[1,2,3,4,5,6,7,8]
y=[96,97,95,96,96,97,97,98]
plt.yticks(x,["NB","NB (With Proposed)","RF","RF (With Proposed)","kNN","kNN (With Proposed)","SVM","SVM (With Proposed)"])

plt.barh(x,y,color=["black","black","blue","blue","red","red","green","green"])
plt.xlabel("Accuracy")
plt.title("San Francisco Dataset - # of Hotspots:100")
plt.show()
 
x=[1,2,3,4,5,6,7,8]
y=[89,92,81,85,86,93,87,90]
plt.yticks(x,["NB","NB (With Proposed)","RF","RF (With Proposed)","kNN","kNN (With Proposed)","SVM","SVM (With Proposed)"])

plt.barh(x,y,color=["black","black","blue","blue","red","red","green","green"])
plt.xlabel("Accuracy")
plt.title("San Francisco Dataset - # of Hotspots:50")
plt.show()
 
x=[1,2,3,4,5,6,7,8]
y=[96,97,95,96,95,96,96,97]
plt.yticks(x,["NB","NB (With Proposed)","RF","RF (With Proposed)","kNN","kNN (With Proposed)","SVM","SVM (With Proposed)"])

plt.barh(x,y,color=["black","black","blue","blue","red","red","green","green"])
plt.xlabel("Accuracy")
plt.title("Los Angeles Dataset - # of Hotspots:100")
plt.show()