import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


transform = ToTensor()


train_data = datasets.CIFAR10(root="deneme tahtası/Hayvan", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root="deneme tahtası/Hayvan", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
for i, (X_train, y_train) in enumerate(train_data):
    break

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(6*6*16, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1, 16*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
    
torch.manual_seed(101)

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []
for epoch in range(epochs):
    
    trn_crrt = 0
    tst_crrt = 0
    
    for i, (X_train, y_train) in enumerate(train_loader):
        i += 1
        
        y_pred = model(X_train) # not flatten
        loss = criterion(y_pred, y_train)
        predicted = torch.max(y_pred.data, 1)[1]
        
        batch_crr = (predicted == y_train).sum() # true 1   false 0
        trn_crrt += batch_crr
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%500 == 0:
            acc = trn_crrt.item()*100 / (100*i)
            print(f"Epoch {epoch} batch {i} loss {loss.item()} accuracy {acc}")
            
            
    train_losses.append(loss)
    train_correct.append(trn_crrt)
    
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data,1)[1]
            tst_crrt +=  (predicted == y_test).sum()
            
    loss = criterion(y_val, y_test)
    test_losses.append(loss)        
    test_correct.append(tst_crrt)        
            
    
torch.save(model.state_dict(), "C:/Users/HP/Desktop/lab 9/AI and ML/PyTorch/deneme tahtası/hayvan_tanima_cnn.pt")

    






