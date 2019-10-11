import numpy as np
import math
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 

def run_neuron(input, target, W, b, transferFunction):
	n = np.dot(W, input) + b
	a = apply_transfer_function(transferFunction, n)
	error = target - a
	return error

def run_perceptron(inputTab, inputNumber, target, W, b, transferFunction, precision):
	inputVector = np.array([inputTab[0][inputNumber], inputTab[1][inputNumber]])
	# print("inputVector : " + str(inputVector))
	error = run_neuron(inputVector, target, W, b, transferFunction)
	# print("error = " + str(error))
	normalization = find_normalization(transferFunction)
	W = np.around(W + error*inputVector.T*normalization, precision)
	b = np.around(b + error*normalization, precision)
	# print("input #" + str(inputNumber))
	# print("W = " + str(W))
	# print("b = " + str(b))
	return (W, b)

def run_epoch(inputON, inputOFF, W, b, transferFunction, precision):
	# print("////////OFF TERMS /////////")
	for offTerm in range(0, len(inputOFF[0])):
		offTarget = find_off_target(transferFunction)
		(W, b) = run_perceptron(inputOFF, offTerm, offTarget, W, b, transferFunction, precision)
	# print("////////ON TERMS /////////")
	for onTerm in range(0, len(inputON[0])):
		(W, b) = run_perceptron(inputON, onTerm, 1, W, b, transferFunction, precision)
	return (W, b)


def train_nn_perceptron(inputON, inputOFF, nbrEpoch, transferFunction, precision):
	W = np.zeros(len(inputON))
	b = 0.0
	for epoch in range(1, nbrEpoch+1):
		# print("EPOCH : #" + str(epoch))
		(W_old, b_old) = (W, b)
		(W, b) = run_epoch(inputON, inputOFF, W, b, transferFunction, precision)
		# print("END EPOCH #" + str(epoch))
		# print("W_old = " + str(W_old))
		# print("b_old = " + str(b_old))
		# print("W = " + str(W))
		# print("b = " + str(b))
		sameWeight = np.array_equiv(W, W_old)
		sameBias = np.array_equiv(b, b_old)
		# print("Weights equal : " + str(sameWeight))
		# print("Bias equal : " + str(sameBias))
		if (sameWeight and sameBias):
			print("END OF THE PROGRAM : Converged in " + str(epoch) + " epochs.")
			return (W,b)
	print("END OF THE PROGRAM : Did not converged in " + str(nbrEpoch) + " epochs.")
	return (W,b)

def apply_transfer_function(transferFunction, n):
	if transferFunction == "hardlim":
		return hardlim(n)
	elif transferFunction == "hardlims" or transferFunction == "sign":
		return hardlims(n)
	elif transferFunction == "sigmoid":
		return sigmoid(n)
	elif transferFunction == "tanh":
		return np.tanh(n)
	else:
		print("Transfer function unknown")
		sys.exit()

def find_normalization(transferFunction):
	if transferFunction == "hardlim":
		return 1
	elif transferFunction == "hardlims" or transferFunction == "sign":
		return 0.5
	elif transferFunction == "sigmoid":
		return 1
	elif transferFunction == "tanh":
		return 0.5
	else:
		print("Transfer function unknown")
		sys.exit()

def find_off_target(transferFunction):
	if transferFunction == "hardlim":
		return 0
	elif transferFunction == "hardlims" or transferFunction == "sign":
		return -1
	elif transferFunction == "sigmoid":
		return 0
	elif transferFunction == "tanh":
		return -1
	else:
		print("Transfer function unknown")
		sys.exit()

def hardlim(input):
	if (input >= 0):
		return 1.0
	else:
		return 0.0

def hardlims(input):
	if (input >= 0):
		return 1.0
	else:
		return -1.0

def sigmoid(n):
  return 1 / (1 + math.exp(-n))
