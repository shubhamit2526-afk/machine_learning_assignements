import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv(r"C:\EXCEL DATA\Classification_2.csv")


x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

dataset.isna().sum()

#applying standardscaler to minimkize the value
#suppose x_train =20000 converts x_train_scaled =0.25665

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

#applying Logistic Regression()

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)


#Appplying Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
cm

#Aplying Accuracy Score


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
ac


#classification Report

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
cr

#ADD BIAS and VARIANCE TO CHECK MODEL ACCURACY

bias = classifier.score(x_train, y_train)
bias

variance = classifier.score(x_train,y_train)
variance






#---------FUTURE PREDICTION-----------------


dataset1 = pd.read_csv(r"C:\EXCEL DATA\Classification_3.csv")

d2 = dataset1.copy()

#d2

dataset1 = dataset1.iloc[:,[2,3]].values

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()

M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()

d2 ['y_pred'] =classifier.predict(M)


#APPLYING KNN

from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier()

classifier_knn.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
y_pred_knn = classifier_knn.predict(x_test)


#APPLYING CONFUSION MATRIX

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred_knn)
confusion

#CHEKING ACCURACY

from sklearn.metrics import accuracy_score
ac_knn = accuracy_score(y_test,y_pred_knn)
ac_knn



d2.to_csv('final1.csv')

# to get the path

import os
os.getcwd()






























