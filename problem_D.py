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
SVM = True
dataset = data.data3
dataset.generate_circle_input(20, 4, 3)
# dataset.print_input()
# dataset.print_label()
dataset.print_excel("data.xls", "test", dataset.input, dataset.label)

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
	W = svm.train_nn_svm(dataset.input, dataset.label)
	print("W = ")
	print(W)
