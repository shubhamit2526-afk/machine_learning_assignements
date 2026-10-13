
#LINEAR REGRESSION
#CALIFORNIA HOUSING PRICE PREDICTION USING LINEAR REGRESSIONS
#DATASET--CALIFORNIA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\EXCEL DATA\Housing1.csv")
df.head()

df.info()

df.describe()

df.isna().sum()

'''
# Dealing with missing values 

df['total_bedrooms'] = pd.to_numeric(df['total_bedrooms'],errors='coerce')
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
df.isna().sum() '''


#Non numeric columns
df['mainroad'] = df['mainroad'].astype('category')
df['guestroom'] = df['guestroom'].astype('category')
df['basement'] = df['basement'].astype('category')
df['hotwaterheating'] = df['hotwaterheating'].astype('category')
df['airconditioning'] = df['airconditioning'].astype('category')
df['prefarea'] = df['prefarea'].astype('category')
df['furnishingstatus'] = df['furnishingstatus'].astype('category')
df.dtypes




#NORMALIZING DATA
#LOG TRANSFORMATION USING NUMPY

df['bedrooms'] = np.log(df['bedrooms']+1)
df['bathrooms'] = np.log(df['bathrooms']+1)
df['area'] = np.log(df['area']+1)
df['stories'] = np.log(df['stories']+1)
#df['bathrooms_per_household'] = np.log(df['bathrooms_per_household']+1)
#df['bedrooms_per_rooms'] = np.log(df['bedrooms_per_rooms']+1)
df['price'] = np.log1p(df['price'])
#df['stories_per_household'] = np.log(df['stories_per_household']+1)






#Sorting the columns now
#by using feature Enginweering
# creating new columns

#df['bathrooms_per_household'] =df['bathrooms']/df['bedrooms']
#df['bedrooms_per_rooms'] = df['bedrooms']/df['bathrooms']
#df['stories_per_household'] = df['stories'] / df['parking']

df['area_per_bedroom'] = df['area'] / df['bedrooms']

df['area_per_bathroom'] = df['area'] / df['bathrooms']

df['total_rooms'] = df['bedrooms'] + df['bathrooms']

df['parking_per_story'] = df['parking'] / (df['stories'] + 1)


df.head()



#CHECKING CORRELATIONS BY HEATMAP PLOT
#Heatmap plot

plt.figure(figsize=(10,6))
sns.heatmap((df.drop(['mainroad','guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'furnishingstatus','prefarea'],axis=1)).corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')


#VISUALIZING TARGET VARIABLE
#HISTOGRAM


plt.figure(figsize = (6,6))
plt.hist(df['price'],bins = 50, edgecolor='black');
plt.title('Distribution of House Prices')
plt.xlabel('Median House Value')
plt.ylabel('Frequency');


#SKEWNESS

#PLOTTING HISTOGRAM ON ALL DATA
#as we see rthe data is right skewed

df.hist(figsize=(16,16))



##CHECKING AGAIN


df.hist(figsize=(16,16))

#HANDLING THE CATEGORICAL DATA
print(df.dtypes)

df['mainroad'].value_counts()
df['guestroom'].value_counts()
df['basement'].value_counts()
df['hotwaterheating'].value_counts()
df['airconditioning'].value_counts()
df['prefarea'].value_counts()
df['furnishingstatus'].value_counts()

#ONE HOT ENCODING DATA

df = pd.get_dummies(df,drop_first=True)
df.head()

df.columns
#CHECKING CORRELATION WITH DUMMY VARIABLES

plt.figure(figsize =(10,8))
sns.heatmap(df.corr(),annot = True,cmap="Blues")
plt.title('Correlation Matrix')


#CHEKING CORRELATION WITH THE AREA AND PROPERTY
#Properties closer to ocean havehigher median values
print(df.dtypes)

plt.figure(figsize=(15,9))
sns.scatterplot(x='area', y='stories', data=df, hue='price',palette='coolwarm')


#Preparing for Data Modeling
#by using standard scaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(df.dtypes)

x = df.drop(['price', ], axis=1)
y = df['price']

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,random_state=42)


'''
scaler = StandardScaler()
x_train_scaled =scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

'''


'''
print(np.isinf(x_train).sum())
print(np.isinf(x_train).sum().sum())

for col in x_train.columns:
    count = np.isinf(x_train[col]).sum()
    if count > 0:
        print(col, count)


cols_with_inf = x_train.columns[np.isinf(x_train).any()]
print(cols_with_inf)

print(np.isinf(x_train).sum().sum())
print(x_train.isnull().sum().sum())
print(x_train.shape)
'''

#Random Forest

from sklearn.ensemble import RandomForestRegressor 

model = RandomForestRegressor(n_estimators=500,
                              max_depth=10,
                              min_samples_split=5,
                              min_samples_leaf=2,
                              random_state=42
                              )


model.fit(x_train,y_train)
model.score(x_test,y_test)
y_train_pred = model.predict(x_train)
y_test_pred=model.predict(x_test)

''' commenying out because not using standard scaler
model.fit(x_train_scaled,y_train)


model.score(x_test_scaled,y_test)

y_train_pred =model.predict(x_train_scaled)
y_test_pred = model.predict(x_test_scaled)
'''

#calculatin R2 score
from sklearn.metrics import mean_squared_error, r2_score





#EXTRACTING RMSE,R2

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nRandom Forest Matrix :")
print(f"Train RMSE : {train_rmse:2f}")
print(f"Test RMSE : {test_rmse:2f}")
print(f"Train R2 : {train_r2:2f}")
print(f"test R2: {test_r2:2f}")


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

grid_search.fit(x_train,y_train)



best_rf = grid_search.best_estimator_
best_rf.score(x_test,y_test)

print("Best Params:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

print(feature_importance.head(10))


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gbr.fit(x_train, y_train)

print("R²:", gbr.score(x_test, y_test))

from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(x_train, y_train)

print("R2:", xgb.score(x_test, y_test))

from sklearn.ensemble import ExtraTreesRegressor

etr = ExtraTreesRegressor(
    n_estimators=500,
    random_state=42
)

etr.fit(x_train, y_train)

print("R2:", etr.score(x_test, y_test))


from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    gbr,
    x,
    y,
    cv=5,
    scoring='r2'
)

print(scores.mean())

print(feature_importance.tail(10))

print(np.isinf(x).sum().sum())
print(x.isnull().sum().sum())

print(scores)
