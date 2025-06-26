import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=1)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


plt.scatter(x, y, color =  'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff ((Linear regression))')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


plt.scatter(x, y, color =  'red')
plt.plot(x, lin_reg_2.predict(x_poly), color = 'blue')
plt.title('Truth or Bluff ((Polynomial regression))')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

print(lin_reg.coef_)

print(lin_reg_2.coef_)
