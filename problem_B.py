import numpy as np
import functions_perceptron as per
import functions_svm as svm
import sys


inputOFF = np.array([[1.0, 2.0, 1.0, 2.0, 1.5],
					 [1.0, 1.0, 2.0, 2.0, 1.5]	])
inputON = np.array([[4.0, 4.0, 5.0, 5.0, 4.5],
					[4.0, 5.0, 4.0, 5.0, 4.5]	])

input_mat = np.array([	[1, 1], [2, 1], [1, 2], [2, 2], [1.5, 1.5],
						[4, 4],	[5, 4],	[4, 5],	[5, 5],	[4.5, 4.5]	])

labels = np.array([ -1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
NBR_EPOCH_MAX = 2000
PERCEPTRON = True
SVM = True

#--- MAIN
print("Problem B")
if PERCEPTRON:
	print("Perceptron learning rule: ")
	(W, b) = per.train_nn_perceptron(inputON, inputOFF, NBR_EPOCH_MAX, "tanh", 1)
	W = np.append(W, b)
	print("W = ")
	print(W)

if SVM:
	print("Support Vector Machine: ")
	W = np.array([0.0, 0.0])
	b = 0.0
	W = svm.train_nn_svm(W, b, input_mat, labels, NBR_EPOCH_MAX)
	print("W = ")
	print(W)
