#coding: utf-8

import os
import torch
import shutil
import datetime
#from datetime import datetime
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models.model import Net
from collections import OrderedDict
from torch.optim.lr_scheduler import *
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

cudnn.benchmark = True

data_path = './trainingData820'
save_model_path = './weights'

parser = argparse.ArgumentParser('Classify images')
parser.add_argument("--batch_size", '-b', type = int, default = 256, help= 'Batch size')
parser.add_argument("--epoch", '-e', type = int, default = 1000, help= 'Epoch')
parser.add_argument("--lr", '-l', type = float, default = 0.001, help= 'Learning rate')
parser.add_argument("--optimizer", '-o', type = str, default = 'Adam', help= 'Adam or SGD')
parser.add_argument('--gpus', '-g',type=str, default='0', help='model prefix')
parser.add_argument('--width', '-wi',type = int, default= 32, help='width')
parser.add_argument('--height', '-he',type = int, default= 32, help='height')
parser.add_argument('--resume_weights', '-r',type = str, default= None, help='Resume weights')
args = parser.parse_args()

if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
f = open(os.path.join(save_model_path, 'train.log'), 'a+', encoding = 'utf-8')
f.write('-'*50 + '\n')
f.write(now + '\n')

multi_gpus = False
if len(args.gpus.split(',')) > 1:
    multi_gpus = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.Resize(size = (args.width, args.height)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=(args.width, args.height)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])  # Imagenet standards
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
    'train': DataLoader(data['train'], batch_size = args.batch_size, shuffle = True, num_workers = 8, pin_memory = True),
    'valid': DataLoader(data['valid'], batch_size = args.batch_size, shuffle = True, num_workers = 8, pin_memory = True),
}

model = Net()
#model = MobileNet(inputs)

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-5

print(model)

'''if args.resume_weights:
    mask_checkpoint = torch.load(args.resume_weights)

    new_mask_state_dict = OrderedDict()

    for k, v in mask_checkpoint.items():
        name = k[7:]  # remove module.
        new_mask_state_dict[name] = v

    model.load_state_dict(new_mask_state_dict)'''

model.cuda()

if multi_gpus:
    model = nn.DataParallel(model).to(device)
else:
    model = model.to(device)

if args.optimizer == 'SGD':
    optimizer=torch.optim.SGD(model.parameters(),lr = args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=WEIGHT_DECAY)
else:
    print('Optimizer value error!')

loss_func = torch.nn.CrossEntropyLoss()

eval_acc_list = []
is_best = False
for epoch in range(args.epoch):

    # training
    train_loss = 0.
    train_acc = 0.
    for inputs, targets in dataloaders['train']:

        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)
        loss = loss_func(predictions, targets)

        train_loss += loss.item()
        train_pred = torch.max(predictions, 1)[1]
        train_correct = (train_pred == targets).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # evaluation
    model.eval()
    eval_loss = 0.
    eval_acc = 0.

    # with torch.no_grad():

    for inputs, targets in dataloaders['valid']:

        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)
        loss = loss_func(predictions, targets)

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
    torch.save(model.state_dict(), os.path.join(save_model_path, '8-24.pth.tar'))
    # 如果是best,则复制最优模型
    if is_best:
        shutil.copyfile(os.path.join(save_model_path, '8-24.pth.tar'),
                        os.path.join(save_model_path, 'best-8-24.pth.tar'))

    # 输出日志信息
    print('epoch {} trainLoss {:.6f} trainAcc {:.6f} validLoss {:.6f} validAcc {:.6f}'.format(
        epoch + 1, train_loss / (len(data['train'])), train_acc / (len(data['train'])), eval_loss / (len(
                data['valid'])), eval_acc / (len(data['valid']))))
    f.write('epoch {} trainLoss {:.6f} trainAcc {:.6f} validLoss {:.6f} validAcc {:.6f}'.format(
        epoch + 1, train_loss / (len(data['train'])), train_acc / (len(data['train'])), eval_loss / (len(
                data['valid'])), eval_acc / (len(data['valid']))) + '\n')

end = datatime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
f.write(end + '\n')
f.close()