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

# Traininng the DTR model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state= 0)
dtr.fit(x, y)

# Predict a new result
# Predict the result
dtr.predict([[6.5]])

# Visualizing the DTR results (higher resolution)

x_grid = np.arange(min(x.values), max(x.values), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color =  'red')
plt.plot(x_grid, dtr.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff - DTR')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()