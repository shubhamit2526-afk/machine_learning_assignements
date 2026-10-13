
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r"C:\EXCEL DATA\Churn_Modelling.csv")

dataset.shape

dataset.isnull().sum()

dataset.drop('RowNumber',axis=1)

dataset.shape

dataset.isnull().sum()

dataset = dataset.drop('RowNumber', axis =1)
dataset
dataset.shape
dataset.isnull().sum()


dataset.Geography.unique()

dataset.Gender.unique()

dataset.Tenure.unique()

dataset.describe()

dataset.info()

dataset.drop(["CustomerId", "Surname"], axis =1, inplace = True)


#Numerical Variables

num_var = [f for f in dataset.columns if dataset[f].dtype!='0']
num_var

#Discrete Variables

dis_var = [f for f in num_var if dataset[f].nunique() < 20]
dis_var

dataset.Tenure.nunique()

#Continuous Variable

cont_var = [f for f in num_var if f not in dis_var]
cont_var

#Categorical Variables

cat_var = [ f for f in dataset.columns if f not in num_var]
cat_var


#EDA

for var in dis_var:
    sns.countplot(dataset[var])
    plt.show()

print(dataset.columns)

dataset.columns = dataset.columns.str.strip()


#Distplot

for var in cont_var:
    sns.distplot(dataset[var], kde=False, color="green")
    plt.show()
    sns.boxplot(dataset[var])
    plt.show()

#CountPlot

for var in cat_var:
    sns.countplot(dataset[var])
    plt.show()


plt.figure(figsize = (20, 10))
sns.heatmap(dataset.corr(),annot=True)


plt.figure(figsize = (20,10))
sns.pairplot(dataset)

#Handling Imbalanced Dataset

exited_yes = dataset[dataset["Exited"]==1]
exited_no = dataset[dataset["Exited"]==0]


print(exited_yes.shape, exited_no.shape)

x = dataset.drop(["Exited"], axis =1)
y = dataset["Exited"]


x.head()
y.head()




cat_1 = pd.get_dummies(dataset["Geography"], drop_first=True)
cat_2 = pd.get_dummies(dataset["Gender"], drop_first=True)

x.head()

x_t = pd.concat([x, cat_1], axis=1)
x_t_f = pd.concat([x_t, cat_2], axis=1)
x_t_f.head()

x_t_f = x_t_f.drop(["Geography", "Gender"], axis=1)

x_t_f.head()
y.head()


from imblearn.over_sampling import SMOTE
oversample = SMOTE(k_neighbors=4)
x_res, y_res = oversample.fit_resample(x_t_f, y)

print(x_res.shape, y_res.shape)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_res = scaler.fit_transform(x_res)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.25)

#Model Selection and Hyperparameter Tuning

params = {
    "learning_rate" : [0.05,0.10,0.15,0.20,0.25,0.30],
    "max_depth" : [3,4,5,6,8,10,12,15],
    "min_child_weight":[1,3,5,7],
    "gamma":[0.0,0.1,0.2,0.3,0.4],
    "colsample_bytree" :[0.3,0.4,0.5,0.7]}

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


from xgboost import XGBClassifier
classifier = XGBClassifier()

random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)
random_search.fit(x_train,y_train)







