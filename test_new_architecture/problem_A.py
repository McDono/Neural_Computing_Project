import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import functions_perceptron as per
import functions_svm as svm
import data
import ConfusionMatrix

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

NBR_EPOCH_MAX = 2000
PERCEPTRON = False
SVM = True
dataset = data.data1

#--- MAIN
print("Problem A")
if PERCEPTRON:
	print("Perceptron learning rule: ")
	(W, b) = per.train_nn_perceptron(dataset.inputON, dataset.inputOFF, NBR_EPOCH_MAX, "hardlims", 1)
	W = np.append(W, b)
	print("W = ")
	print(W)
	per.plot_perceptron(dataset, W)

if SVM:
	# tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}]
	# clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=5, scoring='accuracy')
	# clf.fit(dataset.input, dataset.label)
	# print("Best param: ")
	# print(clf.best_params_)

	svc = svm.run_svm(dataset, cost=1, kernel="linear") #cost value doesn't have any impact here
	svm.plot_svc(svc, dataset.input,dataset.label)
	cm = ConfusionMatrix.ConfusionMatrix(svc, dataset)
	cm.print_matrix()
