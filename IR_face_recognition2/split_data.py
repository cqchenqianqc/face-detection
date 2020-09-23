import os
import shutil
import random
import argparse

parser = argparse.ArgumentParser('split data')
parser.add_argument("--imgPath", '-i',type = str, default=0, help="Images path")
args = parser.parse_args()

PATH = args.imgPath

VALID_RATIO = 0.05
TEST_RATIO = 0.05

files = [x for x in os.listdir(PATH) if not x.startswith('.')]

for file in files:
	os.makedirs(PATH + '/valid/' + file, mode = 0o777)
	os.makedirs(PATH + '/test/' + file, mode = 0o777)
	file_path = os.path.join(PATH,file)
	pictures = os.listdir(file_path)
	num = len(pictures)
	valid_samples = random.sample(pictures, int(VALID_RATIO * num))
	valid = [shutil.move(os.path.join(PATH,file,x),os.path.join(PATH,'valid', file, x)) for x in valid_samples]
	rest = [x for x in pictures if x not in valid_samples]
	test_samples = random.sample(rest,int(TEST_RATIO * num))
	test = [shutil.move(os.path.join(PATH,file,x),os.path.join(PATH,'test', file, x)) for x in test_samples]

	if not os.path.exists(os.path.join(PATH,'train')):
		os.makedirs(PATH + '/train')
	shutil.move(os.path.join(PATH,file),os.path.join(PATH,'train',file))