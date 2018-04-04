# -*- coding: utf-8 -*-
"""
Created on Tue Apr 03 02:07:11 2018

@author: nirmi
"""

from __future__ import division

import pandas as pd
import sklearn.linear_model as lm
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from scipy.stats import ttest_ind
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

######### Data reading and preprocessing ##########
path = "E:\\mgt_6203\\project\\data\\filtered_data_final.csv"
data = pd.read_csv(path)
data["hh_size"].replace('5+', '5', inplace=True)
data["hh_size"] = pd.to_numeric(data["hh_size"], errors='coerce')

data_1 = data[data["quartile"] == 1]
data_2 = data[data["quartile"] == 2]
data_3 = data[data["quartile"] == 3]
data_4 = data[data["quartile"] == 4]

data_list = [data_1, data_2, data_3, data_4]
for d in data_list:
	d.drop(["household_key", "store_id", "income", "quartile", "last_txn_day"], axis=1, inplace=True)

def split_data(x, y, frac=0.3):
	x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,
									 test_size=frac)
	return x_train, x_test, y_train, y_test

###################################################


def explore_data(x, y):

	x_train, x_test, y_train, y_test = split_data(x, y)

	######### Logistic regression #####################################
	logistic_clf = lm.LogisticRegression()
	logistic_model = logistic_clf.fit(x_train, y_train)
	y_pred_logistic = logistic_model.predict(x_test)
	logistic_confusion = metrics.confusion_matrix(y_test, y_pred_logistic)

	print("Confusion matrix and score for logistic regression \n")
	print(logistic_confusion)
	print("\n Score \n")
	print(logistic_model.score(x_test, y_test))

	######### Decision Tree ###########################################
	tree = DecisionTreeClassifier()
	tree.fit(x_train, y_train)
	y_pred_tree = tree.predict(x_test)
	tree_confusion = metrics.confusion_matrix(y_test, y_pred_tree)
	print("\n confusion_matrix for DecisionTree\n")
	print(tree_confusion)

	######### Random Forest ###########################################
	rforest = RandomForestClassifier(n_estimators=15)
	rforest.fit(x_train, y_train)
	y_pred_rforest = rforest.predict(x_test)
	rforest_confusion = metrics.confusion_matrix(y_test, y_pred_rforest)
	print("\n confusion_matrix for RandomForest\n")
	print(rforest_confusion)

	######### Adaboost ################################################
	adaboost = AdaBoostClassifier()
	adaboost.fit(x_train, y_train)
	y_pred_adaboost = rforest.predict(x_test)
	adaboost_confusion = metrics.confusion_matrix(y_test, y_pred_adaboost)
	print("\n confusion_matrix for Adaboost Algorithm\n")
	print(adaboost_confusion)

	######### Support Vector Machines #################################
	svm = SVC(C=10)
	svm.fit(x_train, y_train)
	y_pred_svm = svm.predict(x_test)
	svm_confusion = metrics.confusion_matrix(y_test, y_pred_svm)
	print("\n confusion_matrix for SVM Algorithm\n")
	print(svm_confusion)

	######### Naive Bayes Classifier ##################################
	nb_clf = GaussianNB()
	nb_clf.fit(x_train, y_train)
	y_pred_nb = nb_clf.predict(x_test)
	nb_confusion = metrics.confusion_matrix(y_test, y_pred_nb)
	print("\n confusion_matrix for Naive Bayes Algorithm\n")
	print(nb_confusion)

	######### Feature Ranking #########################################
	clf = ExtraTreesClassifier()
	clf.fit(x, y)
	feature_series = pd.Series(clf.feature_importances_, index = x.columns)
	feature_series.sort_values(ascending=False, inplace=True)
	print("\n Feature importance ranking \n")
	print(feature_series)

	######### PCA #####################################################
	pca = PCA(n_components=2)
	x_pca = pca.fit_transform(x)
	print("\n Explained variance ratio \n")
	print(pca.explained_variance_ratio_)
	colors = ["navy", "darkorange"]
	target_names = ["churn", "retained"]
	plt.figure()
	for color, i, target_name in zip(colors, [0,1], target_names):
		plt.scatter(x_pca[y==i, 0], x_pca[y==i, 1], color=color, label=target_name)
	plt.legend(loc="best", shadow=False, scatterpoints=1)
	plt.title('PCA of customer churn behavior')
	plt.xlabel('PCA 1')
	plt.ylabel('PCA 2')
	plt.show()


###############################################################

y_1 = data_1["if_churn"]
data_1.drop(["if_churn"], axis=1, inplace=True)

explore_data(data_1, y_1)
