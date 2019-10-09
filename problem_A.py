
import numpy as np
import functions as fn
import sys

inputOFF = np.array([[1.0, 2.0, 1.0, 2.0, 1.5],
					 [1.0, 1.0, 2.0, 2.0, 1.5]])
inputON = np.array([[4.0, 4.0, 5.0, 5.0, 4.5],
					[4.0, 5.0, 4.0, 5.0, 4.5]])



NUMBER_EPOCH = 20
PROBLEM_A = True
PROBLEM_B = False

#--- DEBUG

#--- MAIN

if PROBLEM_A:
	print("Problem A")
	W = np.array([0.0, 0.0])
	b = 0.0
	(W, b) = fn.train_neural_network(W, b, inputON, inputOFF, NUMBER_EPOCH, "hardlims", 1)
if PROBLEM_B:
	print("Problem B")
	W = np.array([0.0, 0.0])
	b = 0.0
	(W, b) = fn.train_neural_network(W, b, inputON, inputOFF, NUMBER_EPOCH, "tanh", 1)
	#precision 1 : 6 epochs
	#precision 2 : 26 epochs
	#precision 3 : 1453 epochs
