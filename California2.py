
#LINEAR REGRESSION
#CALIFORNIA HOUSING PRICE PREDICTION USING LINEAR REGRESSIONS
#DATASET--CALIFORNIA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\EXCEL DATA\California_Housing.csv")
df.head()

df.info()

df.describe()

df.isna().sum()


# Dealing with missing values 

df['total_bedrooms'] = pd.to_numeric(df['total_bedrooms'],errors='coerce')
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
df.isna().sum()


#Non numeric columns
df['ocean_proximity'] = df['ocean_proximity'].astype('category')
df.dtypes

#Sorting the columns now
#by using feature Enginweering
# creating new columns

df['rooms_per_household'] =df['total_rooms']/df['households']
df['bedrooms_per_rooms'] = df['total_bedrooms']/df['total_rooms']
df['population_per_household'] = df['population'] / df['households']


df.head()



#CHECKING CORRELATIONS BY HEATMAP PLOT
#Heatmap plot

plt.figure(figsize=(10,6))
sns.heatmap((df.drop(['ocean_proximity'],axis=1)).corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')


#VISUALIZING TARGET VARIABLE
#HISTOGRAM


plt.figure(figsize = (6,6))
plt.hist(df['median_house_value'],bins = 50, edgecolor='black');
plt.title('Distribution of House Prices')
plt.xlabel('Median House Value')
plt.ylabel('Frequency');


#SKEWNESS

#PLOTTING HISTOGRAM ON ALL DATA
#as we see rthe data is right skewed

df.hist(figsize=(16,16))


#NORMALIZING DATA
#LOG TRANSFORMATION USING NUMPY

df['total_rooms'] = np.log(df['total_rooms']+1)
df['total_bedrooms'] = np.log(df['total_bedrooms']+1)
df['population'] = np.log(df['population']+1)
df['households'] = np.log(df['households']+1)
df['rooms_per_household'] = np.log(df['rooms_per_household']+1)
df['bedrooms_per_rooms'] = np.log(df['bedrooms_per_rooms']+1)
df['population_per_household'] = np.log(df['population_per_household']+1)



##CHECKING AGAIN


df.hist(figsize=(16,16))

#HANDLING THE CATEGORICAL DATA

df['ocean_proximity'].value_counts()

#ONE HOT ENCODING DATA

df = pd.get_dummies(df,drop_first=True)
df.head()

#CHECKING CORRELATION WITH DUMMY VARIABLES

plt.figure(figsize =(10,8))
sns.heatmap(df.corr(),annot = True,cmap="Blues")
plt.title('Correlation Matrix')


#CHEKING CORRELATION WITH THE AREA AND PROPERTY
#Properties closer to ocean havehigher median values

plt.figure(figsize=(15,9))
sns.scatterplot(x='latitude', y='longitude', data=df, hue='median_house_value',palette='coolwarm')


#Preparing for Data Modeling
#by using standard scaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


x=df.drop('median_house_value',axis= 1)
y = df['median_house_value']

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,random_state=42)

#transform -->Used only on scale type class for scaling e.g.standardScaler()

scaler = StandardScaler()
x_train_scaled =scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)



#Random Forest

from sklearn.ensemble import RandomForestRegressor 

#fit()-->used to learn the mean and standard deviation
# No--scaler-->no need of Transform

model = RandomForestRegressor()
model.fit(x_train_scaled,y_train)


model.score(x_test_scaled,y_test)

y_train_pred =model.predict(x_train_scaled)
y_test_pred = model.predict(x_test_scaled)


#calculatin R2 score
from sklearn.metrics import mean_squared_error, r2_score





#EXTRACTING RMSE,R2

train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))
train_r2 = r2_score(y_train,y_train_pred)
test_r2 = r2_score(y_test,y_test_pred)

print("\nRandom Forest Matrix :")
print(f"Train RMSE : {train_rmse:2f}")
print(f"Test RMSE : {test_rmse:2f}")
print(f"Train R2 : {train_r2:2f}")
print("test R2: {test_r2:2f}")


#VISUALIZING MOST IMP FUNCTION IN OUR DATASET


#FEATURE IMPORTAMCE


feature_importance = pd.DataFrame({
    'feature':x.columns,
    'importance':model.feature_importances_}).sort_values('importance', ascending=False)


plt.figure(figsize=(10,6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature_importance  - Random Forest')
plt.xlabel('Importance')


#GRIDSEARCH CV

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


param_grid = {
    'n_estimators':[100,200],
    'max_depth':[10,20,None],
    'min_samples_split' :[2,5]
    }


# FGRID SEARCH

model = RandomForestRegressor()
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=1)

grid_search.fit(x_train_scaled,y_train)



best_rf = grid_search.best_estimator_

print(best_rf.score(x_test_scaled,y_test))









