''' 
run  python3 xml_path.py
修改xml文件中path包含中文：导致xml无法正常使用（使用时替换相应的路径）

'''
import xml.dom.minidom
import os

path = '/home/chenq/darknet/shelf-trainingData/bojin/'
sv_path = '/home/chenq/darknet/shelf-trainingData/tt2/'  
files = os.listdir(path)

#遍历xml文件
for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  
    root = dom.documentElement 
    #获取xml中的path的值
    item = root.getElementsByTagName('path') 
    #获取xml中的fold值
    folder1=root.getElementsByTagName('folder')
    a, b = os.path.splitext(xmlFile)
    #修改xml中的path以及folder值
    for i in item:
        i.firstChild.data = '/home/chenq/darknet/JPEGImages' 
    for f in folder1:
        f.firstChild.data='starbuck20200730-bojin'
    #保存修改后的xml文件
    with open(os.path.join(sv_path, xmlFile), 'w') as fh:
        dom.writexml(fh)
   
    
