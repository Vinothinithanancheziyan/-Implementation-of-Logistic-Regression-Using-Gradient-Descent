# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the Logistic Regression Using Gradient Descent

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset and print the values.
3. Define X and Y array and display the value.
4. Find the value for cost and gradient.
5. Plot the decision boundary and predict the Regression value.

## Program:
Program to implement the the Logistic Regression Using Gradient Descent.

Developed by:VINOTHINI T

RegisterNumber: 212223040245

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
def gradient_descent(theta, X, y, alpha, num_iterations):
  m = len(y)
  for i in range(num_iterations):
    h = sigmoid(X.dot(theta))
    gradient = X.T.dot(h - y) / m
    theta -= alpha * gradient
  return theta
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
  h = sigmoid(X.dot(theta))
  y_pred = np.where(h >= 0.5,1, 0)
  return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
```

## Output:
### Dataset
![Image](https://github.com/user-attachments/assets/c1745a6f-e4b5-4e78-9b9f-78788ac59de0)
![Image](https://github.com/user-attachments/assets/f281f0d2-60c6-4b85-8023-38733eb13a08)
![Image](https://github.com/user-attachments/assets/3bae5756-d52f-4192-a2ad-821533364253)
![Image](https://github.com/user-attachments/assets/debd485f-2cd9-449e-a56f-81d0fa55466b)

### Accuracy and Predicted Values
![Image](https://github.com/user-attachments/assets/a30109f4-b4c7-4384-955c-db5c94f6d1a9)
![Image](https://github.com/user-attachments/assets/ed35700e-4e93-4de2-9721-f17b7b1dec89)
![Image](https://github.com/user-attachments/assets/8a1749c6-f59a-4033-a81f-4869b58a4b28)
![Image](https://github.com/user-attachments/assets/54218add-e3bc-4081-9df2-75f790652894)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

