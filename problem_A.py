
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import functions_perceptron as per
import functions_svm as svm
import data

NBR_EPOCH_MAX = 2000
PERCEPTRON = True
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

if SVM:
	print("Support Vector Machine: ")
	W = svm.train_nn_svm(dataset.input, dataset.label)
	print("W = ")
	print(W)
