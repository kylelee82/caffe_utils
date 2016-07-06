# caffe_utils

This repo has the following scripts that can be used in conjunction with Caffe:

1. batch_predict.test.py - Generates a Kaggle submissions for S* Driver competition.  It takes a caffemodel and generates a CSV for submission or ensembling
2. earlystop.py - Early stopping script for K-fold caffe run (train.py)
3. estimate_mc_logloss.py - Estimates log-loss for S* Driver competition based on a submission csv versus a ground truth csv
4. monitor_train.py - Same as earlystop.py, except runs one iteration only
5. visualize.py - Plot a log file from Caffe

