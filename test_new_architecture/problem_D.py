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
from sklearn.model_selection import train_test_split
import functions_svm as svm

NBR_EPOCH_MAX = 100
SVM = True
dataset = data.data3
dataset.generate_circle_input(200, 4, 3)
# dataset.print_input()
# dataset.print_label()

#--- MAIN
print("Problem D")
if SVM:
	# svc = SVC(C=100.0, kernel='rbf', gamma=1)
	# svc.fit(dataset.input, dataset.label)
	# svm.plot_svc(svc, dataset.input, dataset.label)
	# cm = ConfusionMatrix.ConfusionMatrix(svc, dataset)
	# cm.print_matrix()
	X_train, X_test, y_train, y_test = train_test_split(dataset.input, dataset.label, train_size=0.5, random_state=2)

	tuned_parameters = [{'C': [0.01, 0.1, 1, 10, 100],
	'gamma': [0.5, 1,2,3,4]}]
	clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=10, scoring='accuracy')
	clf.fit(X_train, y_train)
	print("Best param: " + str(clf.best_params_))
	print("Best estimator: " + str(clf.best_estimator_))
	bestSVC = clf.best_estimator_
	svc = SVC(C=100.0, kernel='rbf', gamma=1)
	bestSVC.fit(X_train, y_train)
	svm.plot_svc(bestSVC, X_test, y_test)
	cm = ConfusionMatrix.ConfusionMatrix(bestSVC, X_test, y_test)
	cm.print_matrix()
