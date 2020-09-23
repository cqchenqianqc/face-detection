# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:26:20 2020

@author: chenqqian
run python3 batch_video_image.py
解帧视频

"""
import os
import argparse
import numpy as np
from cv2 import *
import glob

#视频的路径
videoPath="/home/chenq/2020-07-27/*.mp4"
videofile=glob.glob(videoPath)
count=0
for i in videofile:
    cap = VideoCapture(i)
    fps=int(cap.get(CAP_PROP_FPS))
    width = cap.get(3)
    height = cap.get(4)
    frameIndex=0
    while cap.isOpened():
        ret,frame = cap.read()
        frameIndex += 1
        if ret:
            if frameIndex % 15 == 0: #将每15帧的视频进行保存
                count+=1
                cv2.imwrite("/home/chenq/2020-07-27/" +'cup_20200729-haiya'+str(count) + '.jpg', frame)
                print(count)
        else:
            break
    print('wangchen')

'''
##改变图像大小
#jpgFile=glob.glob("D:\\data-video\\cup1\\*.jpg")
jpgFile=glob.glob("D:\\data-video\\cup1\\*.jpg")
#
'''
'''#if __name__ == "__main__":       #主函数
    realpath = os.path.realpath(__file__)       #获取当前执行脚本的绝对路径
    dirname = os.path.dirname(realpath)       #去掉文件名，返回目录（realpath的）
    extension = 'jpg'                                        #寻找文件类型：jpg
    file_list = glob.glob('*.'+extension)             #glob.glob 获取当前工作目录下(所有.jpg结尾的文件名称，返回一个列表。）
    filetxt = open(os.path.join(dirname, 'name.txt'), 'w')  打开一个文件，文件绝对路径是  dirname（目录）+name.txt
 c=123
for jpgfile in jpgFile:
    img=cv.imread(jpgfile)
    img_crop=img[0:608,0:368]
    img_new=cv.copyMakeBorder(img_crop,0,0,0,240,cv.BORDER_REPLICATE)
    c=c+1
    cv2.imwrite("D:\\data-video\\cup_608\\" + "cup_20200509"+str(c) + '.jpg', img_new)
    print(c)
'''
