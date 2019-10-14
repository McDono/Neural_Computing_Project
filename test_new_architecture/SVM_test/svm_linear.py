# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


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

np.random.seed(5)
X = np.random.randn(20,2)
y = np.repeat([1,-1], 10)
X[y == -1] = X[y == -1] +1

# plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
# plt.xlabel('X1')
# plt.ylabel('X2')

svc = SVC(C=1, kernel='linear')
svc.fit(X, y)
# plot_svc(svc, X, y)

svc2 = SVC(C=0.1, kernel='linear')
svc2.fit(X, y)
# plot_svc	(svc2, X, y)

tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}]
clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X, y)
# print("Results: ")
# print(clf.cv_results_)
print("Best param: ")
print(clf.best_params_)

np.random.seed(1)
X_test = np.random.randn(20,2)
y_test = np.random.choice([-1,1], 20)
print(X_test)
print(y_test)
X_test[y_test == 1] = X_test[y_test == 1] -1
print(	X_test)

svc2 = SVC(C=0.001, kernel='linear')
svc2.fit(X, y)
y_pred = svc2.predict(X_test)
confusion_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred), index=svc2.classes_, columns=svc2.classes_)
print(confusion_matrix)

X_test[y_test == 1] = X_test[y_test == 1] -1
plt.scatter(X_test[:,0], X_test[:,1], s=70, c=y_test, cmap=mpl.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

svc3 = SVC(C=1, kernel='linear')
svc3.fit(X_test, y_test)
plot_svc(svc3, X_test, y_test)
