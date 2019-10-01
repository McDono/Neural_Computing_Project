
import numpy as np
import functions as fn
import sys

inputOFF = np.array([[1.0, 2.0, 1.0, 2.0, 1.5],
					 [1.0, 1.0, 2.0, 2.0, 1.5]])
inputON = np.array([[4.0, 4.0, 5.0, 5.0, 4.5],
					[4.0, 5.0, 4.0, 5.0, 4.5]])

W = np.array([0.0, 0.0])
b = 0.0

NOMBRE_EPOCH = 6
PROBLEM_A = True
PROBLEM_B = False

#--- DEBUG

#--- MAIN

if PROBLEM_A:
	print("Problem A")
	for epoch in range(1, NOMBRE_EPOCH+1):
		print("EPOCH : #" + str(epoch))
		(W_old, b_old) = (W, b)
		(W, b) = fn.run_epoch(inputON, inputOFF, W, b, "hardlims")
		print("END EPOCH #" + str(epoch))
		print("W_old = " + str(W_old))
		print("b_old = " + str(b_old))
		print("W = " + str(W))
		print("b = " + str(b))
		sameWeight = np.array_equiv(W, W_old)
		sameBias = np.array_equiv(b, b_old)
		# print("Weights equal : " + str(sameWeight))
		# print("Bias equal : " + str(sameBias))
		if (sameWeight and sameBias):
			print("END OF THE PROGRAM : Converged in " + str(epoch) + " epochs.")
			sys.exit()
	print("END OF THE PROGRAM : Did not converged in " + str(NOMBRE_EPOCH) + " epochs.")

if PROBLEM_B:
	print("Problem B")
	for epoch in range(1, NOMBRE_EPOCH+1):
		print("EPOCH : #" + str(epoch))
		(W_old, b_old) = (W, b)
		(W, b) = fn.run_epoch(inputON, inputOFF, W, b, "tanh")
		print("END EPOCH #" + str(epoch))
		print("W_old = " + str(W_old))
		print("b_old = " + str(b_old))
		print("W = " + str(W))
		print("b = " + str(b))
		sameWeight = np.array_equiv(W, W_old)
		sameBias = np.array_equiv(b, b_old)
		# print("Weights equal : " + str(sameWeight))
		# print("Bias equal : " + str(sameBias))
		if (sameWeight and sameBias):
			print("END OF THE PROGRAM : Converged in " + str(epoch) + " epochs.")
			sys.exit()
	print("END OF THE PROGRAM : Did not converged in " + str(NOMBRE_EPOCH) + " epochs.")
