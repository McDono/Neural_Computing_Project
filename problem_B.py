import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
import numpy as np
import functions_perceptron as per
import functions_svm as svm
import data

NBR_EPOCH_MAX = 2000
PERCEPTRON = True
SVM = True
dataset = data.data1

#--- MAIN
print("Problem B")
if PERCEPTRON:
	print("Perceptron learning rule: ")
	(W, b) = per.train_nn_perceptron(dataset.inputON, dataset.inputOFF, NBR_EPOCH_MAX, "tanh", 1)
	W = np.append(W, b)
	print("W = ")
	print(W)

if SVM:
	print("Support Vector Machine: ")
	W = np.array([0.0, 0.0])
	b = 0.0
	W = svm.train_nn_svm(dataset.input, dataset.label, NBR_EPOCH_MAX)
	print("W = ")
	print(W)
