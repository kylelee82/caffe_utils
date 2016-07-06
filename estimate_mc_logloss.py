#!/usr/bin/python

import pandas as pd
import numpy as np
import re
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

# point 1:
# estimated logloss = 0.80950
# LB logloss = 0.76401
lb_logloss_adder  = 0.04549

# point 2 (latest submission):
# estimated logloss = 0.73859
# LB logloss = 0.68487
lb_logloss_adder2 = 0.05372

# estimate submission logloss (for statefarm driver)
ground_truth_file = "driver.Y_test.csv"
#submission_file   = "statefarm_submission.logit_20160504b.lb_0p37027.csv"
#submission_file   = "final_submission.vgg16.csv"
#submission_file   = "fold6.vgg16.csv"
#submission_file   = "statefarm_submission.logit_2CNN_10folds.csv"
#submission_file   = "statefarm_submission.logit.csv_20160626_lb_0p22355_local_0p22411.avg.csv"
#submission_file    = "statefarm_submission.rf.csv"
submission_file = "statefarm_submission.avg.csv"
#submission_file = "fold1.vgg19.csv"


# read target file
ground_truth_csv = pd.read_csv(ground_truth_file)
print("Info: Reading ground truth file = "+str(ground_truth_file))
submission_csv = pd.read_csv(submission_file)
print("Info: Reading submission file = "+str(submission_file))

# set up numpy arrays
ground_truth = np.zeros(shape=(np.size(ground_truth_csv["Image"])))
submission   = np.zeros(shape=(np.size(ground_truth_csv["Image"]),10))
ground_truth_dict = dict()

for index in range(0,np.size(ground_truth_csv["Image"])):
    image = ground_truth_csv["Image"][index]
    image = re.sub('kaggle\/statefarm\/test\/','',image)
    mclass = ground_truth_csv["Class"][index]
    ground_truth_dict[image] = index
    ground_truth[index] = mclass

for index in range(0,np.size(submission_csv["img"])-1):
    image = submission_csv["img"][index]
    if image in ground_truth_dict:
        for j in range(0,10):
            submission[ground_truth_dict[image]][j] = submission_csv["c"+str(j)][index]
        submission[ground_truth_dict[image]] = submission[ground_truth_dict[image]] / sum(submission[ground_truth_dict[image]])
        
score = log_loss(ground_truth,submission)
print("Info: Estimated custom log-loss="+str(score))
print("Info: Estimated LB log-loss="+str(score-lb_logloss_adder))

predicted_class = np.argmax(submission,axis=1)
cm = confusion_matrix(ground_truth, predicted_class)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
accuracy = sum(predicted_class == ground_truth)*100.00/len(ground_truth)
print("Overall Accuracy="+str(float(accuracy))+"%")
print("=================")
print("Per-class accuracy")
for i in range(0,10):
        accuracy = sum(predicted_class[np.where(ground_truth == i)] == i)*100.00/len(predicted_class[np.where(ground_truth == i)])
        print("c"+str(i)+"="+str(float(accuracy))+"%, size="+str(sum(ground_truth == i)))

