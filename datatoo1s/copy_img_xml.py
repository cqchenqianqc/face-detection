#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run  python3 copy_img_xml.py（使用时，须替换成自己的路径）
复制img和xml文件，扩充数据集

"""

import os
import cv2
import numpy as np
import shutil
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    #文件的路径
    input_dir      = '/home/chenq/darknet/shelf-trainingData/starbuckcup_20200827_finished/'
    for root,dirs,files in os.walk(input_dir):
        #print(files)
        j=0
        for file in files:
            fname = os.path.join(root,file)
            f,ext = os.path.splitext(fname)
            path,fn = os.path.split(fname)
            j=j+1
            if ext == '.jpg':
                img         = cv2.imread(fname)
                #print(f)
                # 源文件路径
                xmlFile=f+'.xml'
                #print(xmlFile)
                for i in range(2):
                    cv2.imwrite('/home/chenq/darknet/shelf-trainingData/tt/'+'20200831'+str(j)+'-'+str(i)+'.jpg',img)
                    xmlfile='/home/chenq/darknet/shelf-trainingData/tt/'+'20200831'+str(j)+'-'+str(i)+'.xml'
                    #print(xmlfile)
                    shutil.copy(xmlFile,xmlfile)
                    
                
            
