import numpy as np
# import quadprog
from quadprog import solve_qp

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
	print("# DEBUG: begining quadprog")
	print(P)
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

input_mat = np.array([
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
])
labels = np.array([ -1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
labels_mat = np.diag(labels)
ones_mat = np.ones((len(input_mat), 1))
input_augmented_mat = np.concatenate((input_mat, ones_mat), axis=1)
H = np.identity(len(input_mat[0]) + 1)
f = np.zeros(len(input_mat[0])+1)
# f = np.zeros((len(input_mat[0])+1, 1))
A = - np.dot(labels_mat, input_augmented_mat)
# c = - ones_mat
c = - np.ones(10)
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

W = quadprog_solve_qp(H, f, A, c)
print("W = ")
print(W)
