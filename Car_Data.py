
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score



car_data = pd.read_csv(r"C:\EXCEL DATA\car_data.csv")

car_data.head()


car_data = car_data.drop(['Make'], axis = 1)

car_data['origin']
