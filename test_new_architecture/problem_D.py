import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import functions_perceptron as per
import functions_svm as svm
import data
import plot

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
import functions_svm as svm

NBR_EPOCH_MAX = 100
SVM = False
dataset = data.data3
dataset.generate_circle_input(200, 4, 3)
# dataset.print_input()
# dataset.print_label()
dataset.print_excel("data.xls", "test", dataset.input, dataset.label)

#--- MAIN
print("Problem D")
if SVM:
	# print("Support Vector Machine: ")
	# W = svm.train_nn_svm(dataset.input, dataset.label)
	# print("W = ")
	# print(W)
	svc = SVC(C=100.0, kernel='rbf', gamma=1)
	svc.fit(dataset.input, dataset.label)
	svm.plot_svc(svc, dataset.input, dataset.label)
