'''
run python3 mean_and_std.py 
求出数据集颜色三个通道的mean和std
'''
import os
import numpy as np
import cv2
 
ims_path='/home/chenq/darknet/trainingData20200722-replace/JPEGImages/'
ims_list=os.listdir(ims_path)
R_means=[]
G_means=[]
B_means=[]

B_stds=[]
G_stds=[]
R_stds=[]
for im_list in ims_list:
    im=cv2.imread(ims_path+im_list)
    im = im[0:720, 0:1280] 
    #im=im/255.
#extrect value of diffient channel
    im_B =im[:,:,0]
    im_G =im[:,:,1]
    im_R =im[:,:,2]
#count mean for every channel
    im_B_mean=np.mean(im_B)
    im_G_mean=np.mean(im_G)
    im_R_mean=np.mean(im_R)
    
    im_B_std=np.std(im_B)
    im_G_std=np.std(im_G)
    im_R_std=np.std(im_R)
    
#save single mean value to a set of means
    B_means.append(im_B_mean)
    G_means.append(im_G_mean)
    R_means.append(im_R_mean)
    
    B_stds.append(im_B_std)
    G_stds.append(im_G_std)
    R_stds.append(im_R_std)
    
    
a=[B_means,G_means,R_means]
b=[B_stds,G_stds,R_stds]
mean=[0,0,0]
std=[0,0,0]
#count the sum of different channel means
mean[0]=np.mean(a[0])
mean[1]=np.mean(a[1])
mean[2]=np.mean(a[2])

std[0]=np.mean(b[0])
std[1]=np.mean(b[1])
std[2]=np.mean(b[2])
#print('mean of BGR of data\n[{}，{}，{}]'.format( mean[0],mean[1],mean[2]))
print(mean)
print(std)