import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

#read the dataset
dataset = pd.read_csv('Salary_Data.csv') 
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

#replace all the nan
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[1:2])
X[1:2] = imputer.transform(X[1:2])

#split into the train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print('First Linear Regression')
print(regressor)

#Visualizing the traininng set results
y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title ('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set result
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title ('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()