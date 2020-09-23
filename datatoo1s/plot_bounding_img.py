# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:19:34 2020

@author: chenqian
"""
'''
将xml中框坐标，画入图中
run--  python3 plot_bounding_img.py  --imgPath ./trainingdata/image/  --savePath ./trainingdata/bounding_box/
   --imgPath: path for image and xml 
   --savePath: path for save 
'''

import os
import cv2
import argparse
import shutil
import xml.etree.ElementTree as xml_tree

parser = argparse.ArgumentParser('Show Images!')
parser.add_argument("--imgPath", '-i',type=str, default=None, help="Image Path")
#parser.add_argument("--annPath", '-a',type=str, default=None, help="Annotations Path")
parser.add_argument("--savePath", '-s',type=str, default=None, help="Save Path")
args = parser.parse_args()
if not os.path.exists(args.savePath):
	os.makedirs(args.savePath)

#遍历文件路径
for root, dirs, files in os.walk(args.imgPath):
    for file in files:
        if file.endswith('.jpg'):
            imgpath=os.path.join(root,file)
            img=cv2.imread(imgpath)
            xmlPath = os.path.join(root, file.replace('jpg', 'xml'))
            
            tree = xml_tree.parse(xmlPath)
            rot =tree.getroot()
            objects=rot.findall('object')
            for element in objects:
                bndbox=element.find('bndbox')
                xmin=int(bndbox.find('xmin').text)
                xmax=int(bndbox.find('xmax').text)
                ymin=int(bndbox.find('ymin').text)
                ymax=int(bndbox.find('ymax').text)
                #在图上画边框
                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),2)
            cv2.imwrite(os.path.join(args.savePath,file),img)
			