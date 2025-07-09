#importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


#splitting the dataset into the traininng and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print('----Before Features Scaling------')
print (x_train)
print (y_test)

#feature scaling
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print('----After Features Scaling------')
print (x_test)
print (y_test)

#training the logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state = 0)
regressor.fit(x_train, y_train)

#predicting the new result
y_pred = regressor.predict(sc.transform([[30, 87000]]))
np.set_printoptions(precision=2)

print('Predicting the Test set result-1')
print(y_pred)

#predicting the test set results
y_pred_test = regressor.predict(x_test)
np.set_printoptions(precision=2)
print('Predicting the Test set result-2')
print(np.concatenate((y_pred_test.reshape(len(y_pred_test), 1), y_test.values.reshape(len(y_test), 1)), axis=1))

#making the confusion matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test, y_pred_test)
print('----------Confusion metrics------')
print(cm)

as1 = accuracy_score(y_test, y_pred_test)
print('-----------Accuracy score---------')
print(as1)

#visualizing the traning set results
#visualizing the test set results