import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

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


np.random.seed(8)
X = np.random.randn(200,2)
X[:100] = X[:100] +2
X[101:150] = X[101:150] -2
y = np.concatenate([np.repeat(-1, 150), np.repeat(1,50)])
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=2)
# plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.show()

svm = SVC(C=100.0, kernel='rbf', gamma=1)
svm.fit(X_train, y_train)
plot_svc(svm, X_test, y_test)

tuned_parameters = [{'C': [0.01, 0.1, 1, 10, 100],
'gamma': [0.5, 1,2,3,4]}]
clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X_train, y_train)
# print(clf.best_params_)

plot_svc(clf.best_estimator_, X_test, y_test)
print(confusion_matrix(y_test, clf.best_estimator_.predict(X_test)))
print(clf.best_estimator_.score(X_test, y_test))
