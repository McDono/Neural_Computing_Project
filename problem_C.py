import sys
import numpy as np
import functions_perceptron as per
import functions_svm as svm
import data

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

if SVM:
	print("Support Vector Machine: ")
	W = np.array([0.0, 0.0])
	b = 0.0
	W = svm.train_nn_svm(dataset.input, dataset.label, NBR_EPOCH_MAX)
	print("W = ")
	print(W)