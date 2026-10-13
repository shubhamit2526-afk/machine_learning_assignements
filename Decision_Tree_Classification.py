
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r"C:\EXCEL DATA\Classification_2.csv")

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test =train_test_split(x,y,test_size =0.20,random_state = 0)


#Training the decision tree classification Model

#Tree algorithmsdoes not require feature scaling
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion="entropy", max_depth=3)
classifier.fit(x_train, y_train)

#predicting the test set results

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)



bias = classifier.score(x_train, y_train)
bias

variance = classifier.score(x_test,y_test)
variance

 














