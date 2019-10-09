from sklearn import svm
import numpy as np

# inputOFF = np.array([[1.0, 2.0, 1.0, 2.0, 1.5],
# 					 [1.0, 1.0, 2.0, 2.0, 1.5]])
# inputON = np.array([[4.0, 4.0, 5.0, 5.0, 4.5],
# 					[4.0, 5.0, 4.0, 5.0, 4.5]])

# inputs = np.array([	[1.0, 2.0, 1.0, 2.0, 1.5, 4.0, 4.0, 5.0, 5.0, 4.5],
# 					[1.0, 1.0, 2.0, 2.0, 1.5, 4.0, 5.0, 4.0, 5.0, 4.5]])

inputs = [
	[1, 1],
	[2, 1],
	[1, 2],
	[2, 2],
	[1.5, 1.5],
	[4, 4],
	[5, 4],
	[4, 5],
	[5, 5],
	[4.5, 4.5]
]

labels = [ -1, -1, -1, -1, -1, 1, 1, 1, 1, 1]

def filter_support_vectors(initalSupportVectors):
	supportVectors = [[], []]
	distanceMin = measure_vector_distance(initalSupportVectors[0], initalSupportVectors[1])
	for i in range(len(initalSupportVectors)-1):
		for j in range(i+1, len(initalSupportVectors)):
			distance = measure_vector_distance(initalSupportVectors[i], initalSupportVectors[j])
			if (distance < distanceMin):
				distance = distanceMin
				supportVectors = [initalSupportVectors[i], initalSupportVectors[j]]
	return supportVectors


def measure_vector_distance(V1, V2):
	return abs(np.linalg.norm(V1) - np.linalg.norm(V2))


clf = svm.SVC(gamma='scale')
clf.fit(inputs, labels)
initalSupportVectors = clf.support_vectors_
print("Initial support vectors : ")
print(initalSupportVectors)

supportVectors = filter_support_vectors(initalSupportVectors)
print("True support vectors : " )
print(supportVectors)
