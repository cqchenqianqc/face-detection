# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:41:10 2020

@author: chenqian

run  python3 train_voc_kmeans.py  --filePath  ./trainingData20200722-replace
1.划分数据集（训练集与测试集）
2.创建labels文件夹，并且计算每一个xml文件中box对应的labels，以及训练集和测试集的路径
3.计算训练集anchor box

"""


import os
import random
import argparse

#训练集占数据集为0.8，测试集为0.2
train_percent = 0.8  

parser = argparse.ArgumentParser('train_test!')
parser.add_argument("--filePath", '-i',type=str, default=None, help="Image Path")
parser.add_argument('--k', type=int, help = 'number of cluster', default = 9)
parser.add_argument('--size',type=int, default=768, help="labels Path")
parser.add_argument('--plotPath',type=str, default='/home/chenq/darknet/picture/', help="plot Path")

args = parser.parse_args()
xmlpath=args.filePath+'xmlcup/'

#遍历xml数据集
total_xml = os.listdir(xmlpath)

#数据集的数量
num=len(total_xml)
list=range(num)
tv=int(num*train_percent)

#随机从数据集中挑选数据
train= random.sample(list,tv)
 
ftrainpath=args.filePath+'train'+'.txt'
ftestpath=args.filePath+'test'+'.txt'
ftrain=open(ftrainpath,'w')
ftest=open(ftestpath,'w')

#将挑选好的测试集以及训练集写到train.txt和test.txt
for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in train:
        ftrain.write(name)
    else:
        ftest.write(name)

ftrain.close()
ftest .close()

print("wangcheng+train_test")


#根据xml生成对应的labels
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import argparse


sets=[('cup', 'train'), ('cup', 'test')]   
#classes=["cup"]
#classes=["green_01","yellow_01","blue","red_01","white","green_02","yellow_02","red_02","orange","black","gray","layer"]
classes=["cup","layer"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

#定义convert_annotation函数：进行labels转变
def convert_annotation(year, image_id):
    in_filepath=args.filePath+'xmlcup/'+'%s.xml'%(image_id)
    in_file=open(in_filepath)
    #in_file = open('./trainingDatam/xmlcup/%s.xml'%(image_id))   
    #out_file = open('./trainingDatam/labels/%s.txt'%(image_id), 'w')
    out_filepath=args.filePath+'labels/'+'%s.txt'%(image_id)
    out_file=open(out_filepath,'w')

    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

#修改train.txt和test.txt中的文件名的路径
for year, image_set in sets:
    image_idspath=args.filePath+'%s.txt'%(image_set)
    image_ids = open(image_idspath).read().strip().split()

    list_filepath=args.filePath+'txt'+'%s_%s.txt'%(year, image_set)
    list_file = open(list_filepath,'w')
    for image_id in image_ids:
        list_file.writepath='/home/chenq/darknet/'+args.filePath+'JPEGImages/'+'%s.jpg'%(image_id)
        list_file.write(list_file.writepath+'\n')
        convert_annotation(year, image_id)
    list_file.close()   
    
print('wangcheng+train_rename')



#进行数据集中anchor box 生成
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys

fd_label=args.filePath+'labels/'
ncluters=args.k
plotpath=args.plotPath


n_class     =     len(classes)
hout=args.size
wout=args.size
wh          = np.array([])
 
for root in os.listdir(fd_label):
    fn_label = os.path.join(fd_label,root)
    f,ext    = os.path.splitext(fn_label)
    if ext == ".txt":
        a = np.loadtxt(fn_label)
        if wh.shape[0] == 0:
            wh = a
            if len(wh.shape) == 1:
                wh = np.expand_dims(wh,axis = 0)
        else:
            if len(a.shape) == 1:
                a = np.expand_dims(a,axis = 0)
            wh = np.concatenate((wh,a),axis=0)
    
wh0 = wh[:, 3:]
wh0[:,0] = wh0[:,0] * wout/32
wh0[:,1] = wh0[:,1] * hout/32

wh_res = KMeans(n_clusters = ncluters, random_state=0).fit(wh0)

pathfile1=plotpath+'anchortest.png'
plt.figure(figsize = (12, 12))
plt.scatter(wh0[:, 0], wh0[:, 1], c = wh_res.labels_)
plt.title("cluster result")
plt.savefig(pathfile1)
result=wh_res.cluster_centers_
result=result*32

print(result)
result = result[np.argsort(result[:,0])]
str_anchors = ''
for i,v in enumerate(result):
    if i == (args.k - 1):
        str_anchors += ' ' + str(int(v[0] + 0.5)) + ',' + str(int(v[1] + 0.5))
    else:
        str_anchors += ' ' + str(int(v[0] + 0.5)) + ',' + str(int(v[1] + 0.5)) + ','
print(str_anchors)
mse =[]
anchor_list = []
for n_c in range(2,10):
    wh_res = KMeans(n_clusters=n_c, random_state=0).fit(wh0)
    mse.append(wh_res.inertia_)
    anchor_list.append(wh_res.cluster_centers_)

pathfile2=plotpath+'anchort.png'
plt.figure(figsize=(12, 12))
plt.plot(range(2,10),mse)
plt.xlabel('n_cluster')
plt.ylabel('mse')
plt.savefig(pathfile2)
print(np.mean(wh0,axis=0))
    
    

    
    
    
