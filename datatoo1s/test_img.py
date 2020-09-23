'''  
run   python3 test_img.py
通过test.txt文件，生成对应的test图片文件夹
'''
import os
import shutil
image_ids = open('./test.txt').read().strip().split()
for image_id in image_ids:
    file=os.path.join("./labels/",image_id+'.txt')
    file1=os.path.join("./JPEGImages/",image_id+'.jpg')
    shutil.copy(file1, "/home/chenq/darknet/trainingData20200722-replace/test/")
    