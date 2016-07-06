#!/usr/bin/python

import pandas as pd
from matplotlib.pyplot import *
import os
import math
import sys

############# INPUTS ############
mode = "best_val" # best_val or best_acc
log = "resnet50_train_fold1.log"
snapshot_prefix = "./models/statefarm/snapshot_fold1"
# snapshot rounding
rounding = 2500
# number of iterations before giving up
patience = 25000
#################################

os.system("/media/hdd/caffe/tools/extra/parse_log.py "+log+" .")

train_log = pd.read_csv(log+'.train')
test_log = pd.read_csv(log+'.test')
#train_log.dropna(inplace=True)
#test_log.dropna(inplace=True)
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
	print("Info: Lowest test loss {} ".format(str(min_loss)))
	print("Info: Lowest test loss iteration {} / {} ".format(str(min_iter), str(total_iter)))	
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

	print("Info: Upper bound - Iter: {}, Loss: {}".format(upper,upper_loss))
	print("Info: Lower bound - Iter: {}, Loss: {}".format(lower,lower_loss))
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
        print("Info: Highest test acc {} ".format(str(max_acc)))
        print("Info: Highest test acc iteration {} / {} ".format(str(min_iter), str(total_iter)))
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
        print("Info: Upper bound - Iter: {}, Acc: {}".format(upper,upper_acc))
        print("Info: Lower bound - Iter: {}, Acc: {}".format(lower,lower_acc))
        if upper_acc > lower_acc:
                best_iter = upper
        else:
                best_iter = lower
else:
	print('ERROR: Mode '+mode+' is not supported at the moment!')
	sys.exit(999)

if best_iter > -1:
	best_weight = snapshot_prefix+"_iter_"+str(int(best_iter))+".caffemodel"
	if os.path.isfile(best_weight):
		print("Info: Best weight = "+best_weight)
		os.system("rm -f "+snapshot_prefix+"."+mode+".caffemodel")
		print("Info: Best weight soft-linked to "+snapshot_prefix+"."+mode+".caffemodel")
		os.system("ln -s "+os.getcwd()+"/"+best_weight+" "+os.getcwd()+"/"+snapshot_prefix+"."+mode+".caffemodel")
	else:
		print("ERROR: Cannot find file "+best_weight)

if total_iter > -1 and min_iter > -1:
	if patience < (total_iter - min_iter):
		print("Info: Delta of {} exceeds patience {}".format((total_iter-min_iter),patience))
		# kill caffe job
		# os.system("pkill caffe")
	else:
		print("Info: Delta of {} is still within patience {}".format((total_iter-min_iter),patience))
