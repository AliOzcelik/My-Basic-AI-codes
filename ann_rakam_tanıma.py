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
# from torch.utils import make_grid

transform = ToTensor()
#train_path = r"C:/Users/HP/Downloads/archive(27)/train/train"
#train_data = DataLoader(torchvision.datasets.ImageFolder(train_path,transform=transform), batch_size = 32, shuffle = True)
#train_data = DataLoader(train_path, batch_size = 100, shuffle = True)
#image, label = train_data[0]

train_data = datasets.MNIST(root="deneme tahtası/Digits", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="deneme tahtası/Digits", train=False, download=True, transform=transform)

image, label = train_data[0]

# plt.imshow(image.reshape((28,28)))
# plt.show()
torch.manual_seed(101)

train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))

#for images, labels in train_loader: # 6.000 / 100 batch = 60 times
 #   break

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        #self.out = nn.Linear(32,10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim = 1)
    
    
model = Model() 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

train_losses = []
test_losses = []
train_correct = []
test_correct = []
epochs = 10
for epoch in range(epochs):
    
    trn_crrt = 0
    tst_crrt = 0
    
    for i, (X_train, y_train) in enumerate(train_loader):
        i += 1
        
        y_pred = model(X_train.view(100,-1))
        loss = criterion(y_pred, y_train)
        predicted = torch.max(y_pred.data, 1)[1]
        
        batch_crr = (predicted == y_train).sum()
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
            y_val = model(X_test.view(100, -1))
            predicted = torch.max(y_val.data,1)[1]
            tst_crrt +=  (predicted == y_test).sum()
            
    loss = criterion(y_val, y_test)
    test_losses.append(loss)        
    test_correct.append(tst_crrt)        
            
    
torch.save(model.state_dict(), "C:/Users/HP/Desktop/lab 9/AI and ML/PyTorch/deneme tahtası/rakam_tanima.pt")

