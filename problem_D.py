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
NBR_POINT = 1000
SVM = True
dataset = data.data3
dataset.generate_circle_input(NBR_POINT, 4, 3)
# dataset.print_input()
# dataset.print_label()

optimalParameter = True #determine if the optimal value for gamma and cost are used (True)
						#or if personnalized values are used (False)

#-------PARAMETERS TO MODIFY (works only if optimalParameter = False-------
cost = 10 #optimal value: 10
gamma = 0.5 #optimal value: 0.5

#--- MAIN
print("Problem D")
if SVM:
	print("Number of input data: " + str(NBR_POINT))
	X_train, X_test, y_train, y_test = train_test_split(dataset.input, dataset.label, train_size=0.5, random_state=2)
	if optimalParameter:
		svc = svm.run_svm_nonlinear(X_train, y_train, kernel="rbf")
	else:
		svc = svm.run_svm_nonlinear(X_train, y_train, kernel="rbf",cost=cost,gamma=gamma)
	svm.plot_svc(svc, X_test, y_test)

	cm = ConfusionMatrix.ConfusionMatrix(svc, X_test, y_test)
	print("Support Vector Machine: ")
	cm.print_matrix()
