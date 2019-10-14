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
	svc = SVC(C=1, kernel='linear')
	svc.fit(dataset.input, dataset.label)
	svm.plot_svc(svc, dataset.input,dataset.label)
