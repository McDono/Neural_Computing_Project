
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import functions_perceptron as per
import functions_svm as svm
import data


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.svm import SVC

NBR_EPOCH_MAX = 2000
PERCEPTRON = False
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
	plt.scatter(dataset.input[:,0], dataset.input[:,1], s=70, c=dataset.label, cmap=mpl.cm.Paired)
	pad = 0.25
	x_min, x_max = dataset.input[:, 0].min()-pad, dataset.input[:, 0].max()+pad
	x = np.linspace(x_min,x_max,100)
	y = -W[0]/W[1]*x-b/W[1]
	plt.plot(x, y, '-r')
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.grid()
	plt.show()

if SVM:
	# print("Support Vector Machine: ")
	# W = svm.train_nn_svm(dataset.input, dataset.label)
	# print("W = ")
	# print(W)
	# plt.scatter(dataset.input[:,0], dataset.input[:,1], s=70, c=dataset.label, cmap=mpl.cm.Paired)
	# pad = 0.25
	# x_min, x_max = dataset.input[:, 0].min()-pad, dataset.input[:, 0].max()+pad
	# x = np.linspace(x_min,x_max,100)
	# y = -W[0]/W[1]*x-W[-1]/W[1]
	# plt.plot(x, y, '-r')
	# plt.xlabel('X1')
	# plt.ylabel('X2')
	# plt.grid()
	# plt.show()
	svc = SVC(C=1, kernel='linear')
	svc.fit(dataset.input, dataset.label)
	svm.plot_svc(svc, dataset.input,dataset.label)
