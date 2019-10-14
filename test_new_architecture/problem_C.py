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

NBR_EPOCH_MAX = 100
PERCEPTRON = True
SVM = True
dataset = data.data2

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
	svc = SVC(C=1e5, kernel='linear') #see optimized C
	svc.fit(dataset.input, dataset.label)
	svm.plot_svc(svc, dataset.input,dataset.label)
	cm = ConfusionMatrix.ConfusionMatrix(svc, dataset)
	cm.print_matrix()
