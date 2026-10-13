#HOUSING PRICE PREDICTION 

#Import required libraries and supress warnings

import warnings
warnings.filterwarnings('ignore')



import numpy as np
import pandas as pd
import matplotlib

#Use non interactive backend for matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#In case only plot is imported in some environments
plt.switch_backend('Agg')

%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



#Metrics

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Set seaborn style for plots

sns.set(style='whitegrid')

#for reproducibility

RANDOM_STATE = 42


#Data Loading 

df = pd.read_csv(r"C:\EXCEL DATA\Housing_Data.csv",encoding='ascii', delimiter = ',')

#Quick overview of dataframes

print('Dataframes columns and types :')
print(df.dtypes.head(10))

print(df.dtypes)
#check for missing values

missing_values = df.isnull().sum()
print("\nMissing values per columns")
print(missing_values[missing_values>0])

#Exploratory Data Analysis

#Visualizing dataset from multiplr perspective

#Extract only numeric columns for correlation analysis

numeric_df = df.select_dtypes(include=[np.number])

#check if we have atleast 4 numeric colimns for  meaningful heatmap correlation 

if numeric_df.shape[1] >=4:
    plt.figure(figsize =(12,10))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation heatmap of numeric features')
    plt.tight_layout()
    plt.show()
else:
    print("there is no any correlation finding")
    
#pair plot of few key variables to inspect relationships

selected_columns = ['SalePrice', 'Overall Qual', 'Gr Liv Area', 'Total Bsmt SF']
sns.pairplot(df[selected_columns])
plt.suptitle('pair plot of Selected Features', y=1.02)    
plt.show()  


#Histogram of Saleprice Distribution

plt.figure(figsize=(8,4))
sns.histplot(df['SalePrice'], kde=True,bins=30)  
plt.title('Distribution of Sales')    
plt.xlabel('SalePrice')
plt.show()    

#Bar plot : Average Saleprice by overall qualification

plt.figure(figsize=(8,4))
sns.barplot(x='Overall Qual', y='SalePrice', data=df, estimator=np.mean, ci=None)
plt.title('Average Saleprice by Overall Quality')
plt.show()


#Box plot : Saleprice vs House Age

plt.figure(figsize =(8,4))
sns.boxplot(x='HouseAge', y="SalePrice", data=df)
plt.title('SAlePrice vs HouseAge')
plt.show()





#Prepare Data for Modeling

#We will predict the saleprice using subset of numeric features.
#In real scenario

#We might perform feature engineering, cross validation , selection to enhance model performance

#Define the features (dropping target variable saleprice and non productive columns such as Order)


predictors= numeric_df.drop(columns=['SalePrice', 'Order'], errors='ignore').columns.tolist()

# Define X and Y for regressions

x = df[predictors]
y=df['SalePrice']


#split the data into train and test sets

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=RANDOM_STATE)

#Initialize an train  linear regression model

model = LinearRegression() 
model.fit(x_train, y_train)

#predict on test data

y_pred = model.predict(x_test)

# Calculate Prediction accuracy metrics

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error',mse)
print('R^2 Score',r2)


#plot true vs predicted saleprice

plt.figure(figsize =(8,4))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted SalePrice')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.show()

#StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
y_train = scaler.transform(x_test)


#Creating Evaluation model

def evaluate_model(model):
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    mae = mean_absolute_error(y_test,y_pred)
    
    mse = mean_squared_error(y_test, y_pred)
    
    rmse = np.sqrt(mse)
    
    r2 = r2_score(y_test, y_pred)
    
    print("MAE :",mae)
    print("MSE :",mse)
    print("RMSE :",rmse)
    print("R2 score :",r2)

#LinearRegression

evaluate_model(model)

#Ridge Regression

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
evaluate_model(ridge)

#Lasso Regression

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
evaluate_model(lasso)

# ElasticNet--Combination of L1+L2

from sklearn.linear_model import ElasticNet
elastic = ElasticNet()
evaluate_model(elastic)

#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
    )

evaluate_model(poly_model)


#SVR--Support Vector Regressor

from sklearn.svm import SVR
svr = SVR(kernel='rbf')
evaluate_model(svr)



#Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)
evaluate_model(dt)

#Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators = 100,
    random_state=42
    )

evaluate_model(rf)


from sklearn.linear_model import LinearRegression
import joblib

model = LinearRegression()

model.fit(x_train, y_train)

joblib.dump(model, 'house_price_model.pkl')









