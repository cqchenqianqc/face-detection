##加载并标准化数据集
#coding: utf-8
import torch
import torchvision
#import torchvision.transforms as transforms
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from models.model import Net
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import shutil
import datetime
import time
'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ("green","yellow","blue","red","white","orange","black","gray")

###可视化部分训练的数据


# 输出图像的函数


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机获取训练图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 显示图片
imshow(torchvision.utils.make_grid(images))
# 打印图片标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))'''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


data_path = './data-831'
save_model_path = './weights'

classes = ("green","yellow","blue","red","white","orange","black","gray")

now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
f = open(os.path.join(save_model_path, 'train.log'), 'a+', encoding = 'utf-8')
f.write('-'*50 + '\n')
f.write(now + '\n')

image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.Resize(size = (32, 32)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.226, 0.226, 0.226])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.226, 0.226, 0.226])  # Imagenet standards
    ]),
}

# Datasets from folders,BGR
data = {
    'train':
    datasets.ImageFolder(root = os.path.join(data_path, 'train'),
                         transform=image_transforms['train']),
    'valid':
    datasets.ImageFolder(root = os.path.join(data_path, 'valid'),
                         transform=image_transforms['valid'])
}


# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(data['train'], batch_size = 64, shuffle = True, num_workers = 8, pin_memory = True),
    'valid': DataLoader(data['valid'], batch_size =64, shuffle = True, num_workers = 8, pin_memory = True),
}

net = Net()
print(net)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-5)

epoches=300

eval_acc_list = []
is_best = False
###训练网络
for epoch in range(epoches):
    train_loss = 0.
    train_acc = 0.
    for inputs, targets in dataloaders['train']:
    #get the inputs
        inputs = inputs.to(device)
        targets=targets.to(device)
    #zero the parameter gradients
        
    
    #forward+backward+optimizer
        outputs=net(inputs)
        loss=criterion(outputs,targets)
        train_loss += loss.item()
        train_pred = torch.max(outputs, 1)[1]
        
        
        train_correct = (train_pred == targets).sum()
        train_acc += train_correct.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    '''#print statisics
        running_loss+=loss.item()
        if epoch%10==9:
            print(print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10)))
            running_loss=0.0'''

    net.eval()
    eval_loss = 0.
    eval_acc = 0.
    for inputs, targets in dataloaders['valid']:

        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = net(inputs)
        loss = criterion(predictions, targets)

        eval_loss += loss.item()
        eval_pred = torch.max(predictions, 1)[1]
        num_correct = (eval_pred == targets).sum()
        eval_acc += num_correct.item()

    eval_acc_list.append(eval_acc)
    if eval_acc == max(eval_acc_list):
        is_best = True
        best_pred = eval_acc / (len(data['valid']))
        print('The best best_predAcc is {:.6f}, epoch {}'.format(best_pred, epoch + 1))
        f.write('The best best_predAcc is {:.6f}, epoch {}'.format(best_pred, epoch + 1) + '\n')
    else:
        is_best = False

    # 保存模型
    torch.save(net.state_dict(), os.path.join(save_model_path, 'model-20200904-2.pth.tar'))
    # 如果是best,则复制最优模型
    if is_best:
        shutil.copyfile(os.path.join(save_model_path, 'model-20200904-2.pth.tar'),
                        os.path.join(save_model_path, 'best_model-20200904-2.pth.tar'))

    # 输出日志信息
    print('epoch {} trainLoss {:.6f} trainAcc {:.6f} validLoss {:.6f} validAcc {:.6f}'.format(
        epoch + 1, train_loss / (len(data['train'])), train_acc / (len(data['train'])), eval_loss / (len(
                data['valid'])), eval_acc / (len(data['valid']))))
    f.write('epoch {} trainLoss {:.6f} trainAcc {:.6f} validLoss {:.6f} validAcc {:.6f}'.format(
        epoch + 1, train_loss / (len(data['train'])), train_acc / (len(data['train'])), eval_loss / (len(
                data['valid'])), eval_acc / (len(data['valid']))) + '\n')

end = time.time()
f.write(end + '\n')
f.close()
    

