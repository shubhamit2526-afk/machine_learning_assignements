
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r"C:\EXCEL DATA\Classification_2.csv")

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state =0)


#FEATURE SCALING USING NORMALIZER

# NORMALIZATION --> ROW BY ROW
# STANDARDIZATION --> COLUMN BY COLUMN OR ON FEATURE DATA

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Training the Naive Bayes model on Training set

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(x_train, y_train)

#APPLYING GUASSIANNB NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
classifier_GB = GaussianNB()
classifier_GB.fit(x_train, y_train)


# APLYING MULTINOMIAL NAIVE BAYES

from sklearn.naive_bayes import MultinomialNB
classifier_MB = MultinomialNB()
classifier_MB.fit(x_train, y_train)


# Pedicting the test set results

y_pred = classifier.predict(x_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(x_train, y_train)
bias

variance = classifier.score(x_train, y_train)
variance


























