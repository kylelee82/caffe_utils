#!/usr/bin/python

import pandas as pd
from matplotlib.pyplot import *
import os

os.system("/media/hdd/caffe/tools/extra/parse_log.py resnet50_train_fold1.log .")

train_log = pd.read_csv('./resnet50_train_fold1.log.train')
test_log = pd.read_csv('./resnet50_train_fold1.log.test')
train_log.dropna(inplace=True)
test_log.dropna(inplace=True)

_, ax1 = subplots(figsize=(15,10))
ax2 = ax1.twinx()
ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4)
ax1.plot(test_log["NumIters"], test_log["loss"], 'g')
ax2.plot(test_log["NumIters"], test_log["accuracy"], 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('test_loss')
ax2.set_ylabel('test acuracy')

ax1.set_ylim([0,1])
ax2.set_ylim([0.8,1.0])

show()

