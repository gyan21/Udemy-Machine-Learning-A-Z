import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[: , 1:3])
x[: , 1:3] = imputer.transform(x[: , 1:3])
print('Updated Features: ', x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder= 'passthrough')
x = np.array(ct.fit_transform(x))

print(x)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)


