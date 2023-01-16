import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F

path = r"C:\Users\HP\Desktop\lab 9\AI and ML\PyTorch\deneme tahtası\Moda"
transform = ToTensor()

train_data = torchvision.datasets.FashionMNIST(root=path,train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root=path,train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

for i, (X_train, y_train) in enumerate(train_data):
    break

class_names = train_data.classes
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)    
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

torch.manual_seed(101)
model = CNN()
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_losses = []
test_losses = []
train_correct = []
test_correct = []
epochs = 5

for epoch in range(epochs):
    trn_crrt = 0
    tst_crrt = 0
    for i,(X_train,y_train) in enumerate(train_loader):
        # forward propogation
        i += 1
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        predicted = torch.max(y_pred.data, 1)[1]
        
        batch_crr = (predicted == y_train).sum() # true 1   false 0
        trn_crrt += batch_crr
        
        # back propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%200 == 0:
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
            
    
torch.save(model.state_dict(), "C:/Users/HP/Desktop/lab 9/AI and ML/PyTorch/deneme tahtası/moda_tanima_cnn.pt")

    
"""

class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
        
"""
