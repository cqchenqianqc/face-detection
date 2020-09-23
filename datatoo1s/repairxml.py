'''
run  python3 rapairxml.py
将xml中类别layer即货架的box，修改其大小
'''


import os
import cv2
import numpy as np
import shutil
import xml.etree.ElementTree as ET
#xml的路径
xmlpath='/home/chenq/image-starbuck20200710_finished/'

for root, dirs, files in os.walk('%s'%(xmlpath)):
	for file in files:
		root='/home/chenq/image-starbuck20200710_finished/'
		fname = os.path.join(root,file)
		#print(fname)
		#print(fname)
		t,ext = os.path.splitext(fname)
		if ext == ".xml":
			tree   = ET.parse(fname)
			root   = tree.getroot()
			for obj in root.iter('object'):
				if obj.find('name').text=='layer':
					xmlbox = obj.find('bndbox')
                    #将类别layer的box 向左右各扩大两厘米
					xmin = int(xmlbox.find('xmin').text)-2
					xmax = int(xmlbox.find('xmax').text)+2
					xmlbox.find('xmin').text =  str(xmin)
					xmlbox.find('xmax').text =  str(xmax)
			tree.write(fname, encoding="utf-8")
			print('wangcheng')

