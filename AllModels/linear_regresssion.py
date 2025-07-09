
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state= 0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
