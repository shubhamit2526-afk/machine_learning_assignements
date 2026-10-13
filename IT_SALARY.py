# In polynomial analysis changing degree and getting diffrent values is most imp
# try with 3,4,5,6,7 degrees and predict the best suitable value


%matplotlib inline
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

salary_data = pd.read_csv(r"C:\EXCEL DATA\IT_SALARY_DATA.csv")

# Independent variable
x = salary_data.iloc[:,1:2].values

#DEPENDENT VARIABLE
y = salary_data.iloc[:,2].values

#LINEAR_MODEL --LINEAR ALGORITHM(DEGREE-1))

from sklearn.linear_model import LinearRegression

#saving linearRegression() module into lin_reg object for calling purpose
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#ploynomial model (bydefault degree-2)

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 6)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly, y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#linear Regresssion Visulaizations

x_sorted = np.sort(x, axis = 0)


plt.scatter(x,y, color='red')

plt.plot(
    x_sorted.ravel(),
    lin_reg.predict(x_sorted), 
    color ='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()

# Polynomail Visulaizations

x_grid = np.arange(x.min(), x.max(), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x,y, color = 'green')
plt.plot(
         x_grid.ravel(),
         lin_reg_2.predict(poly_reg.fit_transform(x_grid)),
         color = 'blue'
         )
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()


#Prediction


lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred


#polynomial Prediction

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred
 

print("program exceuted successfully")















