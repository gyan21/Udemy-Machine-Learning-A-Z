#import the the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load the data
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

print(x)
print(y)
y = y.values.reshape(len(y), 1)
print('Y - after the reshape')
print(y)
#Apply the feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler() 
x = sc_x.fit_transform(x)

sc_y = StandardScaler() 
y = sc_y.fit_transform(y)

print('Dataset after features scaling')
print(x)
print(y)

#Traininng the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y.ravel())

#Predict the result
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))

#Visualize the SVR result
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color =  'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff ((Scalar vector regression))')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color =  'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff ((Scalar vector regression - High Resolution))')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()