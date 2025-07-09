import pandas as pd 
import numpy as np  

#Step-1 - Importing and preprocessinng a dataset for ML
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

#Step-2 - Handling missing data in a dataset for ML
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 0:3])
X[:, 0:3] = imputer.transform(X[:, 0:3])

#Step-3 - Encoding categorical data for ML
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

#Step-4 - Dataset splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print('Dataset after features splitting')
print(X_train)
print(X_test)
print(y_train)
print(y_test)

#Step 5 - Train the MLR on traininng set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print('Dataset after Linear Regresssion')
print(X_train)
print(X_test)

#Step-6-Predict the Test set result
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

print('Predicting the Test set result')
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


