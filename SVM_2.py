import numpy as np
import quadprog
from quadprog import solve_qp

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -numpy.vstack([A, G]).T
        qp_b = -numpy.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

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
ones = np.ones((len(input_mat), 1))
input_augmented_mat = np.concatenate((input_mat, ones), axis=1)
H = np.identity(len(input_mat[0]))
f = np.zeros(len(input_mat[0]) + 1)
A = - np.dot(labels_mat, input_augmented_mat)
C = - ones
print("Y = ")
print(labels_mat)
print("X = ")
print(input_mat)
print("ONES = ")
print(ones)
print("X1 = ")
print(input_augmented_mat)
print("H = ")
print(H)
print("f = ")
print(f)
print("A = ")
print(A)
print("C = ")
print(C)

W = quadprog_solve_qp(H, f, A, C)
print(W)
