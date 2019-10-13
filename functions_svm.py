import numpy as np
from quadprog import solve_qp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

DEBUG = True

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

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
	qp_G = P
	qp_a = -q
	if A is not None:
		if A.ndim == 1:
			A = A.reshape((1, A.shape[0]))
		if G is None:
			qp_C = -A.T
			qp_b = -b
		else:
			qp_C = -np.vstack([A, G]).T
			qp_b = -np.hstack([b, h])
		meq = A.shape[0]
	else:  # no equality constraint
		qp_C = -G.T if G is not None else None
		qp_b = -h if h is not None else None
		meq = 0
	solution = solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
	return solution

def train_nn_svm(input_mat, labels):
	W = np.zeros(len(input_mat[0]))
	b = 0.0
	labels_mat = np.diag(labels)
	ones_mat = np.ones((len(input_mat), 1))
	input_augmented_mat = np.concatenate((input_mat, ones_mat), axis=1)
	H = np.identity(len(input_mat[0]) + 1)
	f = np.zeros(len(input_mat[0])+1)
	# f = np.zeros((len(input_mat[0])+1, 1))
	A = - np.dot(labels_mat, input_augmented_mat)
	c = - np.ones(len(input_mat))
	W = quadprog_solve_qp(H, f, A, c)
	W = np.around(W, 4)
	if DEBUG:
		print("Y = ")
		print(labels_mat)
		print("X = ")
		print(input_mat)
		print("ONES = ")
		print(ones_mat)
		print("X1 = ")
		print(input_augmented_mat)
		print("H = ")
		print(H)
		print("f = ")
		print(f)
		print("A = ")
		print(A)
		print("C = ")
		print(c)

	return W
