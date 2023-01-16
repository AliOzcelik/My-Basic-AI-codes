from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("seattle-weather.csv")

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

X = (X - np.min(X))/(np.max(X)-np.min(X))   

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, shuffle = False,test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
