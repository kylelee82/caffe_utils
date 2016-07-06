#!/usr/bin/python
import os
import sys
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import numpy as np
import re
import subprocess
import logging
import datetime

now = datetime.datetime.now()
logging.basicConfig(level=logging.DEBUG, filename="train_8fold."+str(now.strftime("%Y-%m-%d-%H-%M"))+".log", filemode="a+",
	format="%(asctime)-15s %(levelname)-8s %(message)s")

nfolds = 8
master_solver = "./models/statefarm/solver.prototxt"
# use the same random state as keras (vgg19/vgg16)
random_state = 20
last_fold = 2

def data_prep(train_ix,test_ix):
        f = open('../input/driver_imgs_list.csv', 'r')
        g = open('data/statefarm/train.txt', 'w')
        h = open('data/statefarm/test.txt', 'w')
        #test_ix = ['p015','p041','p051','p075']
	logging.info('Train drivers :'+str(train_ix))
	logging.info('Test drivers :'+str(test_ix))
        line = f.readline()
        while 1:
                line = f.readline()
                if line == "":
                        break
                arr = line.strip().split(",")
                if arr[0] in test_ix:
                        h.write('data/statefarm/images/c'+arr[1][-1]+"/"+arr[2]+' '+arr[1][-1]+'\n')
                else:
                        g.write('data/statefarm/images/c'+arr[1][-1]+"/"+arr[2]+' '+arr[1][-1]+'\n')
        f.close()
        g.close()
        h.close()

# first generate the correct fold configuration
driver_list = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 
               'p024', 'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 
               'p047', 'p049', 'p050', 'p051', 'p052', 'p056', 'p061', 
               'p064', 'p066', 'p072', 'p075', 'p081']

kf = KFold(len(driver_list), n_folds=nfolds, shuffle=True, random_state=random_state)
num_fold=0
# rerun data prep generation
for train_drivers, test_drivers in kf:
	logging.info('Starting fold {}...'.format(num_fold))

	if num_fold < last_fold:
		logging.info('Skipping fold {} since it has already been completed!'.format(num_fold))
		num_fold += 1
		continue

	outbuf = open("models/statefarm/solver_fold"+str(num_fold)+".prototxt",'w')
	solver_buf = open(master_solver,'r')
	for line in solver_buf:
		if re.search('snapshot_prefix',line):
			line = re.sub(r"snapshot\"","snapshot_fold"+str(num_fold)+"\"",line)
		outbuf.write(line)
	solver_buf.close()
	outbuf.close()
		
        unique_driver_train = list()
        unique_driver_test = list()
        for i in train_drivers:
                unique_driver_train.append(driver_list[i])
        for i in test_drivers:
                unique_driver_test.append(driver_list[i])
	data_prep(unique_driver_train,unique_driver_test)
	cmd = '/media/hdd/caffe/build/tools/caffe train -solver models/statefarm/solver_fold'+str(num_fold)+'.prototxt -weights models/statefarm/weights.caffemodel 2>&1 | tee resnet50_train_fold'+str(num_fold)+'.log'
	#with open('resnet50_train_fold'+str(num_fold)+'.log','w') as out:
	#	return_code = subprocess.call(cmd, stdout=out)
	#cmd_string = " ".join(cmd)	
	os.system(cmd)
	num_fold += 1
