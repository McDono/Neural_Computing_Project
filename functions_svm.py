import numpy as np
from quadprog import solve_qp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

DEBUG = True

def run_svm_nonlinear(X_train, y_train, kernel, cost=None, gamma=None):
	tuned_parameters = [{'C': [0.01, 0.1, 1, 10, 100],
	'gamma': [0.5, 1,2,3,4]}]
	clf = GridSearchCV(SVC(kernel), tuned_parameters, cv=10, scoring='accuracy')
	clf.fit(X_train, y_train)
	print("Best param: " + str(clf.best_params_))
	print("Best estimator: " + str(clf.best_estimator_))
	if (cost != None and gamma != None):
		svc = SVC(C=cost, kernel=kernel, gamma=gamma)
	else:
		svc = clf.best_estimator_
	svc.fit(X_train, y_train)
	return svc

def run_svm_linear(X_train, y_train, kernel, cost=None):
	tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}]
	clf = GridSearchCV(SVC(kernel), tuned_parameters, cv=2, scoring='accuracy')
	clf.fit(X_train, y_train)
	print("Best param: " + str(clf.best_params_))
	print("Best estimator: " + str(clf.best_estimator_))
	if (cost!=None):
		svc = SVC(C=cost, kernel=kernel)
	else:
		svc = clf.best_estimator_
	svc.fit(X_train, y_train)
	return svc


def plot_svc(svc, X, y, h=0.02, pad=0.25):
	x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
	y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)
	plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
	# Support vectors indicated in plot by vertical lines
	sv = svc.support_vectors_
	plt.scatter(sv[:,0], sv[:,1], c='k', marker='x', s=100, linewidths='1')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()
	print('Number of support vectors: ', svc.support_.size)
