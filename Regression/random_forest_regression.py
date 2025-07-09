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

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)

# Predicting the new result
regressor.predict([[6.5]])

# Visualizing the Random Forest Regression results (higher resolution)

x_grid = np.arange(min(x.values), max(x.values), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color =  'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff - Random forest Regressor')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()