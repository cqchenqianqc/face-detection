''' 
run python3 img_replace.py
从实验室采集的数据，将整个货架裁剪,粘贴到另一个门店背景，这样增大数据量
'''


import cv2
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
import shutil
import argparse

#xmlpath = glob.glob('../test/*.xml')


#定义xybox函数：获得每一个类别的box的坐标
def xybox(filepath):
    a=[]
    in_filepath=open(filepath)
    tree=ET.parse(in_filepath)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        if obj.find('name').text=='shelf':
            #print(obj.find('name').text)
            xmlbox = obj.find('bndbox')
            a.append(int(xmlbox.find('xmin').text))
            a.append(int(xmlbox.find('xmax').text))
            a.append(int(xmlbox.find('ymin').text))
            a.append(int(xmlbox.find('ymax').text))
            a.append(round((a[0]+a[1])/2))
            a.append(round((a[2]+a[3])/2))
    return a

#定义img_enchance函数：实验室采集的数据偏暗，与门店的数据不匹配，因此增加数据的亮度
def img_enchance(image,alpha,beta):
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.15 # Simple contrast control
    beta = 50    # Simple brightness control
    # Initialize values

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    return new_image
count=0
i=0
print(i)

#将实验室中货架的整体移植到门店数据中相同的位置
input_dir='/home/chenq/repalece-image/starbuckcup_20200819_finished-sys/'
for root,dirs,files in os.walk(input_dir):
    for file in files:
        fname = os.path.join(root,file)
        f,ext = os.path.splitext(fname)
        path,fn = os.path.split(fname)
        i=i+1
        
        if ext=='.jpg':
            img=cv2.imread(fname)
            #img=img_enchance(img,1.1,30)
            xmlpath=f+'.xml'
            shelfxy=xybox(xmlpath)
            print(shelfxy)
            h,w,_ = img.shape
            imgx=img[shelfxy[2]:shelfxy[3],shelfxy[0]:shelfxy[1]]
            cv2.imwrite('/home/chenq/repalece-image/shelf/cup_20200819%s'%(i)+'.jpg', imgx)
            src_mask =255 * np.ones(imgx.shape, imgx.dtype)
            #print(imgx.shape)
            center=(shelfxy[4],shelfxy[5])
            print(center)

            imgPath = glob.glob('/home/chenq/repalece-image/starbuck-0817/*.jpg')
            for imgpath in imgPath:
                f1,ext1 = os.path.splitext(imgpath)
                count=count+1
                img2=cv2.imread(imgpath)
                xmlpath2=f1+'.xml'
                shelfxy2=xybox(xmlpath2)
            
                h2,w2,_ = img2.shape
                img2[shelfxy2[2]:shelfxy2[3],shelfxy2[0]:shelfxy2[1]]=144
                cv2.imwrite('/home/chenq/repalece-image/shelf/cup_20200819%s'%(i)+'.jpg', imgx)
                output = cv2.seamlessClone(imgx, img2, src_mask, center, cv2.NORMAL_CLONE)
                cv2.imwrite('/home/chenq/repalece-image/sys-starbuck-0817/cup_20200819-replace%s'%(count)+'.jpg', output)
                newfile='/home/chenq/repalece-image/sys-starbuck-0817/cup_20200819-replace%s'%(count)+'.xml'
                shutil.copy(xmlpath, newfile)
                print(count)

