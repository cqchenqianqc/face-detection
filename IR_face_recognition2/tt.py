#coding: utf-8
from __future__ import division
import os
import cv2
import time
import torch
import shutil
import argparse
import datetime
import torch.nn as nn
import torch.backends.cudnn as cudnn

from PIL import Image
from tqdm import tqdm
from models.mobilenet import MobileNetV2
from collections import OrderedDict
from torch.optim.lr_scheduler import *
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

data_path = './testcolour'


classes=["green","yellow","blue","red","white","orange","black","gray"]

transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
])

model = MobileNetV2()

print(model)

if cpu:
    checkpoint = torch.load('./weights/best_liveness_mobilenetv2.pth.tar', map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load('./weights/best_liveness_mobilenetv2.pth.tar')

new_state_dict = OrderedDict()


for k, v in checkpoint.items():
    name = k[7:]  # remove module.
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

model.eval()

model = model.to(device)

start_time = time.time()

for file in tqdm(os.listdir(data_path)):
    img_path = os.path.join(data_path, file)

    img = cv2.imread(img_path)
    img = Image.fromarray(img)
    frame = transform(img)
    frame = torch.unsqueeze(frame, 0)
    frame = frame.to(device)
    pred = model(frame)
    print(pred)
    label = pred.argmax()
    n=int(label)
    print(classes[n])





