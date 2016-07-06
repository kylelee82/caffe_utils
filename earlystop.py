#!/usr/bin/python

import pandas as pd
from matplotlib.pyplot import *
import os
import math
import sys
import time
import logging
import datetime

########################
### this early stopping script requires a Caffe run to be launched in a 
### a separate terminal, and it will kill caffe processes and increment folds
### when it reads that the iteration - best_iteration > patience
### it will also continuously create a snapshot best_val.caffemodel for each fold
### if the best_val or best_acc criteria is met
########################

############# INPUTS ############
fold_start      = 1  # put non-zero to skip certain folds if you are starting from middle
num_folds       = 8  # 8 = 8-fold
interval        = 15 # checking interval, in minutes
mode            = "best_val" # best_val or best_acc
log_prefix      = "resnet50_train_fold" # prefix with add with fold_num
snapshot_prefix = "./models/statefarm/snapshot_fold" # prefix will add with fold_num
rounding        = 1000 # same as caffe snapshot interval
patience        = 25000 # number of iterations before terminating
#################################

now = datetime.datetime.now()
logging.basicConfig(level=logging.DEBUG, filename="earlystop."+str(now.strftime("%Y-%m-%d-%H-%M-%S"))+".log", filemode="a+",
        format="%(asctime)-15s %(levelname)-8s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

for fold_num in range(num_folds):
	print("---------------------------")
	if fold_num < fold_start:
		logging.info('Skipping fold '+str(fold_num)+' since it is completed...')
		continue
	else:
		logging.info('Start checking fold '+str(fold_num)+' ...')

	while True:
		print("---------------------------")
		now = datetime.datetime.now()
		print("Time : "+str(now.strftime("%Y-%m-%d/%H:%M:%S")))
		log = log_prefix + str(fold_num) + ".log"

		while not os.path.isfile(log):
			logging.info("Log {} does not exist yet, waiting {} minutes".format(log, interval))
			time.sleep(60*interval)

		os.system("/media/hdd/caffe/tools/extra/parse_log.py "+log+" .")

		while not os.path.isfile(log+'.train'):
			logging.info("File {} does not exist yet, waiting {} minutes".format(log+'.train', interval))
			time.sleep(60*interval)
			os.system("/media/hdd/caffe/tools/extra/parse_log.py "+log+" .")

		while not os.path.isfile(log+'.test'):
			logging.info("File {} does not exist yet, waiting {} minutes".format(log+'.test', interval))
			time.sleep(60*interval)
			os.system("/media/hdd/caffe/tools/extra/parse_log.py "+log+" .")

		train_log = pd.read_csv(log+'.train')
		test_log = pd.read_csv(log+'.test')

		best_iter = -1
		total_iter = -1
		min_iter = -1
		if mode is "best_val":
			row=0
			min_loss = 10000
			min_row  = row
			for test_loss in test_log['loss']:
				if test_loss is not None and test_loss < min_loss:
					min_row = row
					min_loss = test_loss
				row+=1
			min_iter = test_log.ix[min_row,'NumIters']
			total_iter = test_log.ix[row-1,'NumIters']
			logging.info("Lowest test loss {} ".format(str(min_loss)))
			logging.info("Lowest test loss iteration {} / {} ".format(str(min_iter), str(total_iter)))	
			upper = math.ceil(min_iter / rounding) * rounding
			lower = math.floor(min_iter / rounding) * rounding
			row = 0
			lower_loss = None
			upper_loss = None
			for iter in test_log['NumIters']:
				if iter >= upper and upper_loss is None:
					upper_loss = test_log.ix[row,'loss']	
				if iter >= lower and lower_loss is None:
					lower_loss = test_log.ix[row,'loss']
				row += 1	

			logging.info("Upper bound - Iter: {}, Loss: {}".format(upper,upper_loss))
			logging.info("Lower bound - Iter: {}, Loss: {}".format(lower,lower_loss))
			if upper_loss > lower_loss:
				best_iter = lower
			else:
				best_iter = upper
		elif mode is "best_acc":
        		row=0
        		max_acc = 0.0
        		max_row  = row
        		for test_acc in test_log['accuracy']:
                		if test_acc is not None and test_acc > max_acc:
                        		max_row = row
                        		max_acc = test_acc
                		row+=1
        		min_iter = test_log.ix[max_row,'NumIters']
        		total_iter = test_log.ix[row-1,'NumIters']
        		logging.info("Highest test acc {} ".format(str(max_acc)))
        		logging.info("Highest test acc iteration {} / {} ".format(str(min_iter), str(total_iter)))
        		upper = math.ceil(min_iter / rounding) * rounding
        		lower = math.floor(min_iter / rounding) * rounding
        		row = 0
        		lower_acc = None
        		upper_acc = None
        		for iter in test_log['NumIters']:
                		if iter >= upper and upper_acc is None:
                        		upper_acc = test_log.ix[row,'accuracy']
                		if iter >= lower and lower_acc is None:
                        		lower_acc = test_log.ix[row,'accuracy']
                		row += 1
        		logging.info("Upper bound - Iter: {}, Acc: {}".format(upper,upper_acc))
        		logging.info("Lower bound - Iter: {}, Acc: {}".format(lower,lower_acc))
        		if upper_acc > lower_acc:
                		best_iter = upper
        		else:
                		best_iter = lower
		else:
			print('ERROR: Mode '+mode+' is not supported at the moment!')
			sys.exit(999)

		if best_iter > -1:
			best_weight = snapshot_prefix+str(fold_num)+"_iter_"+str(int(best_iter))+".caffemodel"
			if os.path.isfile(best_weight):
				logging.info("Best weight = "+best_weight)
				os.system("rm -f "+snapshot_prefix+str(fold_num)+"."+mode+".caffemodel")
				logging.info("Best weight soft-linked to "+snapshot_prefix+str(fold_num)+"."+mode+".caffemodel")
				os.system("ln -s "+os.getcwd()+"/"+best_weight+" "+os.getcwd()+"/"+snapshot_prefix+str(fold_num)+"."+mode+".caffemodel")
			else:
				logging.warning("Cannot find file "+best_weight)

		if total_iter > -1 and min_iter > -1:
			if patience < (total_iter - min_iter):
				logging.info("Delta of {} exceeds patience {}".format((total_iter-min_iter),patience))
				# kill caffe job
				os.system("pkill caffe")
				break
			else:
				logging.info("Delta of {} is still within patience {}".format((total_iter-min_iter),patience))

		logging.info("Sleeping for {} minutes before checking again...".format(interval))
		time.sleep(60*interval)

