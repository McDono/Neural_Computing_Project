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

NBR_EPOCH_MAX = 100
PERCEPTRON = True
SVM = True
dataset = data.data2

optimalParameter = False #determine if the optimal value for cost is used (True)
						#or if personnalized value is used (False)

#-------PARAMETERS TO MODIFY (works only if optimalParameter = False-------
cost = 1 #optimal value: 1

#--- MAIN
print("Problem C")
if PERCEPTRON:
	print("Perceptron learning rule: ")
	(W, b) = per.train_nn_perceptron(dataset.inputON, dataset.inputOFF, NBR_EPOCH_MAX, "hardlim", 1)
	W = np.append(W, b)
	print("W = ")
	print(W)
	per.plot_perceptron(dataset, W)

if SVM:
	if (optimalParameter):
		svc = svm.run_svm_linear(dataset.input, dataset.label , kernel="linear")
	else:
		svc = svm.run_svm_linear(dataset.input, dataset.label , kernel="linear", cost=1)
	svm.plot_svc(svc,dataset.input,dataset.label)
	cm = ConfusionMatrix.ConfusionMatrix(svc, dataset.input,dataset.label)
	cm.print_matrix()
