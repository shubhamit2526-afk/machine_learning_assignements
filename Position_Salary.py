#NON_LINEAR DATA FOR ANALYSIS 

#POSITION SALARY DATA

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#USE NON INTERACTIVE BACKEND FOR MATPLOTLIB

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

%matplotlib inline

import seaborn as sns


#Models Used for Non_linear analysis 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



#Metrics

from sklearn.metrics import r2_score, mean_absolute_error


#Loading Dataset
df = pd.read_csv(r"C:\EXCEL DATA\Position_Salaries.csv")

print(df.head())

#Understanding Dataset

print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())


#Data Cleaning 
#This datast is already clear

df = df.drop("Positon",axis=1)
print(df.head())


#EDA techniques

#Scatter plot

plt.scatter(df["Level"], df["Salary"])
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Level VS Salary")
plt.show()

#Feature Selection

x = df[["Level"]]

y = df[["Salary"]]


#Train_Test_Split Data

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)


#Feature Scaling
#important for SVR & KNN

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)



y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))


#TRAIN MODELS

# LINEAR REGRESSION

lr = LinearRegression()

lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)

print("Linear Regression R2 score :",
      r2_score(y_test, y_pred_lr))




#Plynomial Regression

poly  = PolynomialFeatures()

x_poly_train = poly.fit_transform(x_train)
x_poly_test = poly.transform(x_test)



#Train Models

poly_model = LinearRegression()

poly_model.fit(x_poly_train,y_train)

y_pred_poly = poly_model.predict(x_poly_test)

print("Polynomial Regression R2 score :",
      r2_score(y_test, y_pred_poly))



## Deccision Tree Regrwssion

dt = DecisionTreeRegressor(random_state=42)

dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)

print("Decision Tree R2 score :",
      r2_score(y_test, y_pred_dt))



#Random Forest Regression
#used to calculate multiple  decison tree models anaylisng best one


rf = RandomForestRegressor(
     n_estimators = 100,
     random_state= 42
     )

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print("Random Forest R2 score :",
       r2_score(y_test,y_pred_rf))





#SVR--
#Used to calculate Distance and deimensions
#tries to fit the bst boundary or curve

svr = SVR(kernel='rbf')

svr.fit(x_train_scaled,y_train_scaled.ravel())

#Predict
y_pred_svr = svr.predict(x_test_scaled)

#invese transform

y_pred_svr = scaler_y.inverse_transform(
    y_pred_svr.reshape(-1,1))

print("SVR R2 score :",
      r2_score(y_test,y_pred_svr))


#KNN Regression
# KNN is distnace based

knn = KNeighborsRegressor(n_neighbors=3)

knn.fit(x_train_scaled, y_train.values.ravel())

y_pred_knn = knn.predict(x_test_scaled)

print("KNN R2 score :",
      r2_score(y_test,y_pred_knn))



#MOST IMP PART 
#COMPARING MODELS 

models = {
    "LinearRegression" : y_pred_lr,
    "PolynomialRegression" : y_pred_poly,
    "Decision Tree " : y_pred_dt,
    "Random_forest " : y_pred_rf,
    "SVR " : y_pred_svr,
    "KNN " : y_pred_knn 
    }


for name , pred in models.items():
    print("\n", name)
    
    print("R2 score :",
          r2_score(y_test,pred)) 
    
    print("MAE :",
          mean_absolute_error(y_test,pred))




#VISUALIZATIONS 

#PLOYNOMIAL CURVE VISUALIZATION


x_grid = np.arange(min(x.values),
                   max(x.values),
                   0.1)

x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(x,y, color='red')

plt.plot(
    x_grid,
    poly_model.predict(
        poly.transform(x_grid)
        ),
    color = 'blue'
    )

plt.title("polynomial Regression")
plt.xlabel("Level")
plt.ylabel("Salary")

plt.show()


# RANDOM FOREST VISUALIZATION

plt.scatter(x,y, color ='red')

plt.plot(
    x_grid,
    rf.predict(x_grid),
    color ='green'
    )

plt.title("Random Forest Regression")

plt.xlabel("Level")
plt.ylabel("Salary")

plt.show()


#PREDICT NEW VALUE

#PREDICT SALARY FOR LEVEL 6.5

#POLYNOMIAL REGRESSION

salary = poly_model.predict(
    poly.transform([[6.5]])
    )
print("Predicted Salary :", salary)






























