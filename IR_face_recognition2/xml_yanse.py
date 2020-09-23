import cv2
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
import shutil
import argparse


classes=["green","yellow","blue","red","white","orange","black","gray"]

for i in range(8):
    os.mkdir(str(classes[i]))


def xybox(filepath):
    a=[]
    in_filepath=open(filepath)
    tree=ET.parse(in_filepath)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    objects=root.findall('object')
    for element in objects:
        cls = element.find('name').text
        difficult = element.find('difficult').text
        if cls not in classes or int(difficult)==1:
            continue
        xmlbox = element.find('bndbox')
        a.append(int(xmlbox.find('xmin').text))
        a.append(int(xmlbox.find('xmax').text))
        a.append(int(xmlbox.find('ymin').text))
        a.append(int(xmlbox.find('ymax').text))
        cls_id = classes.index(cls)
        a.append(str(cls_id))
    return a



i=0

input_dir='/home/chenq/darknet/trainingDatacolour/'
for root,dirs,files in os.walk(input_dir):
    for file in files:
        fname = os.path.join(root,file)
        f,ext = os.path.splitext(fname)
        path,fn = os.path.splitext(file)
        i=i+1
        count=0
        if ext=='.jpg':
            img=cv2.imread(fname)
            xmlpath=f+'.xml'
            tree = ET.parse(xmlpath)
            rot =tree.getroot()
            objects=rot.findall('object')
            for element in objects:
                count=count+1
                cls = element.find('name').text
                difficult = element.find('difficult').text
                if cls not in classes or int(difficult)==1:
                    continue
                bndbox=element.find('bndbox')
                xmin=int(bndbox.find('xmin').text)
                xmax=int(bndbox.find('xmax').text)
                ymin=int(bndbox.find('ymin').text)
                ymax=int(bndbox.find('ymax').text)
                cls_id = classes.index(cls)
                imgx=img[ymin:ymax,xmin:xmax]
                yansefile='/home/chenq/darknet/trainingData820/'+str(cls_id)+'/'+path+'-%s'%(count)+'.jpg'
                #print(yansefile)
                cv2.imwrite(yansefile,imgx)
         
          
            

