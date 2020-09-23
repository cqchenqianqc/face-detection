'''
run  python3 xml_rename.py
修改已经标注好的xml中类别（替换成相应的路径）
'''

import xml.etree.ElementTree as ET
import numpy as np
import cv2
from shutil import copyfile
import copy
import os
import argparse
parser = argparse.ArgumentParser('Show Images!')
parser.add_argument("--xmlPath", '-i',type=str, default=None, help="Image Path")
#parser.add_argument("--annPath", '-a',type=str, default=None, help="Annotations Path")
parser.add_argument("--savePath", '-s',type=str, default=None, help="Save Path")
args = parser.parse_args()


#定义类renamebox:将类别替换成自己需要的类
def renamebox(filepath,outpath):
	with open(filepath, 'r') as f:
		data = f.read()
		tmp = copy.deepcopy(data)
		#print(tmp)

	for item in ['red_01', 'red_02']:
		tmp = tmp.replace(item, 'red')
	for item in ['yellow_01', 'yellow_02']:
		tmp = tmp.replace(item, 'yellow')
	for item in ['green_01','green_02']:
		tmp = tmp.replace(item, 'green')
	#print(tmp)

	with open(outpath, 'w') as f:
		f.write(tmp)
	

#遍历文件路径，替换所有的xml中的类别
for root, dirs, files in os.walk(args.xmlPath):
	for file in files:
		fname = os.path.join(root,file)
		f,ext = os.path.splitext(fname)
		im_path,fn =  os.path.split(fname)
		if ext=='.xml':
			file_path = os.path.join(args.savePath, fn)
			#print(file_path)
			renamebox(fname,file_path)
	print('wancheng')


