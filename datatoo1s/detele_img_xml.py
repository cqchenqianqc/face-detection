'''
run python3 detele_img_xml.py
找出JPEGImages文件夹中图片与xmlcup中的xml不匹配的文件

'''

import os
import numpy as np

#train.txt文件的路径
image_idspath='./train.txt'

#读出txt文件中所有的文件名
image_ids = open(image_idspath).read().strip().split()
#print(image_ids)
for image_id in image_ids:
    imgpath='/home/chenq/darknet/trainingData20200722-replace/JPEGImages/'+'%s.jpg'%(image_id)
    xmlpath='/home/chenq/darknet/trainingData20200722-replace/xmlcup/'+'%s.xml'%(image_id)
    #打印出txt文件中，即不在imgpath文件夹中，或者不在xmlpath文件夹中
    if os.path.exists(imgpath)==False:
        print('*'*100)
        print('img')
        print(image_id)
        #os.remove(file_path)
    
    if os.path.exists(xmlpath)==False:
        print('*'*100)
        print('xml')
        print(image_id)

