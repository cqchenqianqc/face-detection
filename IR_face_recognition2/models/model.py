import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

class Net(nn.Module):
    def __init__(self,num_classes=8):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        #self.batchnorm1 = nn.BatchNorm2d(num_features=6, affine=False)
        #self.batchnorm2 = nn.BatchNorm2d(num_features=8, affine=False)
        self.pool2 =nn.MaxPool2d(3,2)
        self.conv2 = nn.Conv2d(6, 8, 3)
        #self.conv3=nn.Conv2d(8,4,1)
        self.fc1 = nn.Linear(288, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)
        
        
    def forward(self, x):
        x =self.pool1(F.relu(self.conv1(x)))
        x =self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        #x = self.batchnorm2(F.relu(self.conv2(x)))
        #x=F.relu(self.conv2(x))
        
        #print(x.shape)
        # = x.view(-1, 8*3*3)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = x.view(x.size(0), -1)
        #x =  F.softmax(x,1)
        return x
        
d = torch.rand(1, 3, 32, 32)
m = Net()
o = m(d)

onnx_path = "onnx_model_name.onnx"
torch.onnx.export(m, d, onnx_path)
 
#netron.start(onnx_path)
