import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("seattle-weather.csv")

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# X = (X - np.min(X))/(np.max(X)-np.min(X))   

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, shuffle = False, test_size=0.2)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4,64)
        self.fc2 = nn.Linear(64,120)
        self.fc3 = nn.Linear(120,120)
        self.fc4 = nn.Linear(120,84)
        self.out = nn.Linear(84,5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)
        return x

model = NeuralNetwork()
torch.manual_seed(100)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


epochs = 100
losses = []

for i in range(epochs):
    
    # Forward and get a prediction
    y_pred = model.forward(X_train)
    
    # calculate loss/error
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    
    if (i % 10) == 0:
        print("Epoch {} and loss is {}".format(i,loss))
        
    # backpropogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

correct = 0
with torch.no_grad():
    
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)
    
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        
        if y_val.argmax().item() == y_test[i]:
            correct += 1

acc = correct/len(X_test)
print(acc)
