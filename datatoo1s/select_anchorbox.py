''' 
python select_anchorbox.py
杯子和货架大小不一致，数量也不一样，因此传统的kmeans挑选出来了的anchor box并不能完全代表数据集的分布
'''
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
import pandas as pd

classes=["green","yellow","blue","red","white","orange","black","gray","layer"]
#classes=["cup"]
n_class     =     len(classes)

fd_label="/home/chenq/darknet/trainingData20200722-replace/labels/"
wout,hout=768,768
wh    = np.array([])
wlayer=[]
wcup=[]

#遍历labels文件夹，得到整个数据集类别的坐标
for root in os.listdir(fd_label):
        fn_label = os.path.join(fd_label,root)
        f,ext    = os.path.splitext(fn_label)
        if ext == ".txt":
            a = np.loadtxt(fn_label)
            if wh.shape[0] == 0:
                wh = a
                #wlayer=a[]
               
                if len(wh.shape) == 1:
                    wh = np.expand_dims(wh,axis = 0)
            else:
                if len(a.shape) == 1:
                    a = np.expand_dims(a,axis = 0)
                wh = np.concatenate((wh,a),axis=0)
print(len(wh))               

for i in range(len(wh)):
    if abs(wh[i][0]-8.00)<0.01:
        wlayer.append(wh[i,:])
    else:
        wcup.append(wh[i,:])

wlayer=np.array(wlayer)
wcup=np.array(wcup)

print(np.mean(wlayer[:,3:],axis=0)*wout)
print(np.max(wlayer[:,3:],axis=0)*wout)
print(np.min(wlayer[:,3:],axis=0)*wout)
#print(wlayer[:,3:])
#print(wcup[:,3:])

 
#进行kmeans计算
def anchorbox_kmeans(plotpath,wh,k,i):
    
    wh0 = wh[:, 3:]
    wh0[:,0] = wh0[:,0] * wout/32
    wh0[:,1] = wh0[:,1] * hout/32
    #mm = MinMaxScaler()
    #mm_data = mm.fit_transform(wh0)
    
    wh_res = KMeans(n_clusters = k, random_state=0).fit(wh0)
    quantity = pd.Series(wh_res.labels_).value_counts()
    print("cluster2 number\n", (quantity))
    '''xmin=np.min(wh0,axis=0)
    xmax=np.max(wh0,axis=0)
    print("*"*100)
    print(xmin)
    print(xmax)'''
    result=wh_res.cluster_centers_
    result=result*32
    
    print("*"*100)
    print(result)
    result = result[np.argsort(result[:,0])]
    str_anchors = ''
    for i,v in enumerate(result):
        if i == (k - 1):
            str_anchors += ' ' + str(int(v[0] + 0.5)) + ',' + str(int(v[1] + 0.5))
        else:
            str_anchors += ' ' + str(int(v[0] + 0.5)) + ',' + str(int(v[1] + 0.5)) + ','
    print('*'*100)
    print(str_anchors)
    pathfile1=plotpath+'anchorcup'+'%s'%(str(i))+'png'
    plt.figure(figsize = (12, 12))
    plt.scatter(wh0[:, 0], wh0[:, 1], c = wh_res.labels_)

    plt.title("cup cluster result")
    plt.savefig(pathfile1)
    
    return str_anchors
    
filepath="/home/chenq/darknet/"
#进行全数据集的kemans的计算
wh_anchors=anchorbox_kmeans(filepath,wh,9,1)
#对cup类别进行4类的anchorbox挑选
wlayer_anchors=anchorbox_kmeans(filepath,wlayer,4,2)
#对layer类别进行5类的anchobox挑选
wcup_anchors=anchorbox_kmeans(filepath,wcup,5,3)

