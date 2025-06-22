import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Step-1 - Importing and preprocessinng a dataset for ML
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :3].values  # Features
y = dataset.iloc[:, -1].values  # Target variable

print('Features: ', x)
print('Targets: ', y)

count = 0
for row in x :
    for item in row:
        if pd.isna(item):
            count += 1
print ('Count of Nan', count)

#Step-2 - Handling missing data in a dataset for ML
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[: , 1:3])
x[: , 1:3] = imputer.transform(x[: , 1:3])
print('Updated Features: ', x)


#Step-3 - Encoding categorical data for ML
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder= 'passthrough')
x = np.array(ct.fit_transform(x))

print(x)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

#Step-4 - Dataset splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print('Dataset after features splitting')
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#Step-5 - Features scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.fit_transform(x_test[:, 3:])
print('Dataset after features scaling')
print(x_train)
print(x_test)


