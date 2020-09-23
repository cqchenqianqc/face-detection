#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run python3 repair_xml_1280_arg.py  --imgPath ./trainingData20200817
将图片修改为1280*1280，且相应的改变xml文件
"""

import os
import cv2
import numpy as np
import shutil
import xml.etree.ElementTree as ET
import argparse

#创建del_mkdirs函数：建立相应的文件夹
def del_mkdirs(fd_list):
    for fd in fd_list:
        if os.path.exists(fd):
            shutil.rmtree(fd)
            while os.path.exists(fd): # check if it exists
                pass
            
        os.makedirs(fd)
#建立repair_img_xml函数：填补xml文件
def repair_img_xml(image,in_xml,out_xml,amp_param):
    # step1: repair image
    imgh,imgw,_ = image.shape
         
    tree   = ET.parse(in_xml)
    root   = tree.getroot()
    size   = root.find('size')
    w = int((float(size.find('width').text)))
    h = int((float(size.find('height').text)))
    size.find('height').text = str(imgh)
    size.find('width').text=str(imgw)
    #将xml文件重新写回到文件夹
    tree.write(out_xml, encoding="utf-8")
    
    
#建立remove_bad_file函数：将苹果系统生成的.DS_Store文件删除
def remvove_bad_file(folder):
   for root, dirs, files in os.walk(folder):
        for file in files:
            if file.startswith('._') or file == '.DS_Store':
                fname = os.path.join(root,file)
                print(fname)
                os.remove(fname)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('repair_img_xml!')
    parser.add_argument("--imgPath", '-i',type=str, default=None, help="Image Path")
    args = parser.parse_args()
    input_dir=args.imgPath
    #修补的像素为560
    amp_parameter  = 560
    
    i = 0
    for root,dirs,files in os.walk(input_dir):
        for file in files:
            fname = os.path.join(root,file)
            f,ext = os.path.splitext(fname)
            path,fn = os.path.split(fname)
            
            if ext == '.jpg':
                img         = cv2.imread(fname)
                h1,w1,_=img.shape
                #图片为1280*720（注意cv2读取图片为宽*长：即为720*1280）：
                if h1==720:
                    img_new=cv2.copyMakeBorder(img,0,amp_parameter,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
                #图片为720*1280：
                else:
                    img_new=cv2.copyMakeBorder(img,0,0,0,amp_parameter,cv2.BORDER_CONSTANT,value=[255,255,255])
                
                h,w,_ = img_new.shape
                if h == 1280:
                    i = i + 1
                xml         = f + '.xml'
               
                sav_img     = fname
                sav_xml     = f + '.xml'
                
                img_repair = repair_img_xml(img_new,xml,sav_xml,amp_parameter)
                #保存补白后的图片
                cv2.imwrite(sav_img,img_new)
        
    print('number of image with w=1280:',i)
