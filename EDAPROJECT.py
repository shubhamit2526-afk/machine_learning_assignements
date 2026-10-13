import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Salary=pd.read_csv(r"C:\EXCEL DATA\SalaryData.csv")
print(Salary.shape)
print(Salary.columns)

x = Salary[["Years_of_Experience"]]
y=Salary["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

#===we build MLR model

m=regressor.coef_
m

c=regressor.intercept_
c


#FEATURE SCALING

#STANDARDIZATION

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(x_train)

X_test = sc_X.transform(x_test)

#NORMALIZATION

from sklearn.preprocessing import Normalizer

sc_X = Normalizer()

X_train = sc_X.fit_transform(x_train)

X_test = sc_X.transform(x_test)













