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

def detectcolor(img):
    classes=["green","yellow","blue","red","white","orange","black","gray"]
    cpu = False
    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Image transformations
    transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                                [0.226, 0.226, 0.226])
    ])
    model = MobileNetV2()

# 加载模型
    print(model)

    if cpu:
        checkpoint = torch.load('./weights/best_liveness_mobilenetv2.pth.tar', map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load('./weights/best_liveness_mobilenetv2.pth.tar')

    new_state_dict = OrderedDict()

# 用了nn.DataParallel的模型需要处理才能在cpu上使用
    for k, v in checkpoint.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.to(device)
    start_time = time.time()
        #img = cv2.cvtColor(img,  cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    frame = transform(img)
    frame = torch.unsqueeze(frame, 0)
    frame = frame.to(device)
    pred = model(frame)
    print(pred)
    label = pred.argmax()
    n=int(label)
    label=class[n]
    print(classes[n])
    resume = time.time() - start_time
    print('resume: ', resume)
    images_num = len(os.listdir(data_path))
    print('images num: ', images_num)
    print('FPS: ', images_num / resume)
    
    return label




