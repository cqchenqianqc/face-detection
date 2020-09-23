'''
run python3 cup_colour.python3
将所有的颜色类别改为cup类别（使用时将路径替换为自己的路径）
'''
import os
import copy
from shutil import copyfile

xml_data = '/home/chenq/darknet/trainingData817/xmlcup'
#原本8个颜色类别
kind_list = ['green','yellow','blue','red','white','orange','black','gray']
for root, dirs, files in os.walk('%s'%(xml_data)):
    for file in files:
        fname = os.path.join(root,file)
        f,ext = os.path.splitext(fname)
        im_path,fn =  os.path.split(fname)

        file_path = os.path.join(xml_data, fn)
        #打开xml文件
        with open(fname, 'r') as f:
            data = f.read()
        tmp = copy.deepcopy(data)
        #遍历类别并且替换成cup类别
        for item in kind_list:
            tmp = tmp.replace(item, 'cup')
        #将修改后的xml文件，重新写回到原来的文件
        with open(file_path, 'w') as f:
            f.write(tmp)