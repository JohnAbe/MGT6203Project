# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 21:42:03 2018

@author: nirmi
"""

from __future__ import division
import datetime

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import lightgbm as lgb
from sklearn import svm
import pickle
import calendar
import datetime
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn import metrics


os.chdir('E:\\mgt_6203\\project\\data')


def normalize(data): 
    for i in range(3, data.shape[1]):
        data.iloc[:, i] = np.log(data.iloc[:, i] + 0.00001)
    return data


# data_parent = pd.read_csv('scvo2_features_data_filtered.csv')
data_parent = pd.read_csv('final_data_use.csv')

#data_parent = pd.read_csv('results.csv')

imp_vars = ['total_sales', 'overall_weekly_purchase', 'overall_market_share', 'row_id', 'three_weeks_sales', 'three_weeks_disc', 'three_weeks_freq', 'five_weeks_sales','five_weeks_disc','five_weeks_freq']

#datax = data_parent[['total_sales','total_disc','num_orders','total_qty','overall_weekly_purchase','frac_share','overall_market_share','row_id','three_weeks_sales','three_weeks_disc','three_weeks_freq','five_weeks_sales','five_weeks_disc','five_weeks_freq','weeklag']]
datax = data_parent[imp_vars]

#datax = data_parent[['tissue.extraction','temp','ph','hb','lactate','tissue.extraction-SMA','tissue.extraction-momentum','temp-SMA','temp-momentum','ph-SMA','ph-Momentum','hb-SMA','hb-momentum','lactate-SMA','lactate-momentum']]

datay = data_parent[['if_churn']]
X_train, X_test, y_train, y_test = train_test_split(datax, datay, test_size=0.20, random_state=42)

# xg-boost model
d = 10
w = 8
xgbmodel = xgb.XGBClassifier(silent=False, max_depth= d, learning_rate=0.1, scale_pos_weight = w)
xgbmodel.fit(X_train, np.ravel(y_train.iloc[:,0]), eval_metric = 'auc')
y_pred = pd.DataFrame(xgbmodel.predict(X_test))
accuracy_score(np.ravel(y_test), y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f_score = 2 * precision * recall / (precision + recall)
print(xgbmodel.feature_importances_)
plt.rc('xtick', labelsize=6) 
plt.rc('ytick', labelsize=6)
xgb.plot_importance(xgbmodel)
#pyplot.savefig('destination_path.eps', format='eps', dpi=1000)
plt.show()
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
auc_xgb = metrics.auc(fpr, tpr)


# logistic regression

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
lr_output = logisticRegr.predict(X_test)
accuracy_score(y_test, lr_output)
tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y_test, lr_output).ravel()
recall_lr = tp_lr/(tp_lr+fn_lr)
precision_lr = tp_lr/(tp_lr+fp_lr)
f_score_lr = 2 * precision_lr * recall_lr / (precision_lr + recall_lr)
fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(y_test, lr_output, pos_label=1)
auc_lr = metrics.auc(fpr_lr, tpr_lr)


# random forest 
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(X_train, y_train)
rf_pred = clf.predict(X_test)
accuracy_score(y_test, rf_pred)
tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_test, rf_pred).ravel()
recall_rf = tp_rf/(tp_rf+fn_rf)
precision_rf = tp_rf/(tp_rf+fp_rf)
f_score_rf = 2 * precision_rf * recall_rf / (precision_rf + recall_rf)
fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(y_test, rf_pred, pos_label=1)
auc_rf = metrics.auc(fpr_rf, tpr_rf)
