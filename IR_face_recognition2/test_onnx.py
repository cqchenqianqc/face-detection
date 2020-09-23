# -*- coding: utf-8 -*-
# @Author: Yan An
# @Date: 2020-07-01 15:06:18
# @Last Modified by: Yan An
# @Last Modified time: 2020-07-16 11:08:20
# @Email: an.yan@intellicloud.ai
import os
import cv2
import onnx
import numpy as np
import onnxruntime as rt

from tqdm import tqdm

sess = rt.InferenceSession("./weights/liveness_mobilenetv2_sim.onnx")

fake = 0
fake_fake = 0
real = 0
fake_real = 0

#create input data
path = './data/test/fake'
# path = './anti_image'
for file in tqdm(os.listdir(path)):
	input_data = cv2.imread(os.path.join(path, file))
	input_data = cv2.resize(input_data, (64, 64))
	input_data = np.transpose(input_data, (2, 0, 1))
	input_data = np.expand_dims(input_data, 0)
	input_data = np.float32(input_data)
	input_data = (input_data / 255 - 0.67) / 0.268

	#create runtime session
	# get output name
	input_name = sess.get_inputs()[0].name
	output_name= sess.get_outputs()[0].name
	output_shape = sess.get_outputs()[0].shape
	#forward model
	res = sess.run([output_name], {input_name: input_data})
	out = np.array(res)
	# print(out)

# 	if out[0][0][0] > out[0][0][1]:
# 		fake += 1
# 	else:
# 		fake_fake += 1

# print('fake: ', fake)
# print('fake_fake: ', fake_fake)

	if out[0][0][0] < out[0][0][1]:
		real += 1
	else:
		fake_real += 1

print('real: ', real)
print('fake_real: ', fake_real)



