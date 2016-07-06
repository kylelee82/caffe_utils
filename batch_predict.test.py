#!/usr/bin/python

import sys
import caffe
import os
import glob
import pandas as pd
import datetime
import math

print('Loading model...')
caffe.set_mode_gpu()
model_def = './models/statefarm/model-predict.prototxt'
model_weights = './models/statefarm/snapshot_fold0.best_val.caffemodel'
batch_size = 8

net = caffe.Net(model_def, model_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].reshape(batch_size,3,224,224)

print('Predicting test images...')
path = os.path.join('../input/imgs/test/*.jpg')
files = glob.glob(path)

ids=[]
preds=[]
count=0

thr = math.floor(len(files)/10)
batch_cnt=0
for fl in files:
	print("Reading test image {} number {}".format(fl, count))
	flbase = os.path.basename(fl)
	ids.append(flbase)
	image = caffe.io.load_image(fl)
	transformed_image = transformer.preprocess('data',image)
	net.blobs['data'].data[batch_cnt,...] = transformed_image

	batch_cnt += 1
	if batch_cnt >= batch_size:
		output = net.forward()
		for i in range(batch_size):
			output_prob = output['prob'][i]
			preds.append(list(output_prob))
			print("Output probabilities:" + str(list(output_prob)))
		batch_cnt = 0
	count += 1
	if count % thr == 0:
		print('Predicted {} images from {}'.format(count, len(files)))

print('Complete image predictions!')
print('Creating submission...')
result = pd.DataFrame(preds, columns = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
result.loc[:, 'img'] = pd.Series(ids, index=result.index)
now = datetime.datetime.now()
subm_path = './subm'
if not os.path.isdir(subm_path):
	os.mkdir(subm_path)
sub_file = os.path.join(subm_path,'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv')
result.to_csv(sub_file, index=False)


