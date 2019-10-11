import sys
import numpy as np
import functions_perceptron as per
import functions_svm as svm
import data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

NBR_EPOCH_MAX = 100
PERCEPTRON = False
SVM = False
dataset = data.data3
dataset.generate_circle_input(10, 4, 2)
dataset.print_input()
dataset.print_label()
dataset.print_excel("data.xls", "test3", dataset.output)

#--- MAIN
print("Problem D")
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
