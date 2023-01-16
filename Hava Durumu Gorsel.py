import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image


path = r"C:\Users\HP\Desktop\lab 9\AI and ML\PyTorch\deneme tahtası\Hava Durumu\dataset"
path2 = r"C:\Users\HP\Desktop\lab 9\AI and ML\PyTorch\deneme tahtası\Hava Durumu\test_set"
transform = ToTensor()

train_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
test_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(root=path, transform=train_transform)
test_data = datasets.ImageFolder(root=path2, transform=test_transform)

torch.manual_seed(42)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

class_names = train_data.classes
for i, (images,labels) in enumerate(train_loader):
    break

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.conv3 = nn.Conv2d(16, 26, 5)
        #self.dropout = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(16*13*13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,11)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        #x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        #x = F.relu(self.conv3(x))
        #x = F.max_pool2d(x,2,2)
        #x = self.dropout(x)
        x = x.view(-1, 16*13*13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
model = CNN()    
torch.manual_seed(101)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 3
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
            










