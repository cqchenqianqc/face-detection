import os
import cv2
import argparse
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser('Calculate mean and std')
parser.add_argument("--img_path", '-i', type = str, required = True, default = None, help= 'Image path')
parser.add_argument("--img_width", '-iw', type = int, required = True, default = None, help= 'Image width')
parser.add_argument("--img_height", '-ih', type = int, required = True, default = None, help= 'Image height')
args = parser.parse_args()

means, stdevs = [], []
img_list = []

i = 0
for root, dirs, files in os.walk(args.img_path):
	for file in tqdm(files):
	    img = cv2.imread(os.path.join(root, file))
	    img = cv2.resize(img, (args.img_width ,args.img_height))
	    img = img[:, :, :, np.newaxis]
	    img_list.append(img)
	    i += 1

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
 
# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()

print('RGB: ')
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))