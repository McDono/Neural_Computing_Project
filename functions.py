import numpy as np
import math

def run_neuron(input, target, W, b, transferFunction):
	n = np.dot(W, input) + b
	a = apply_transfer_function(transferFunction, n)
	error = target - a
	return error

def run_perceptron(inputTab, inputNumber, target, W, b, transferFunction):
	inputVector = np.array([inputTab[0][inputNumber], inputTab[1][inputNumber]])
	print("inputVector : " + str(inputVector))
	error = run_neuron(inputVector, target, W, b, transferFunction)
	print("error = " + str(error))
	normalization = findNormalization(transferFunction)
	W = W + error*inputVector.T*normalization
	b = b + error*normalization
	print("input #" + str(inputNumber))
	print("W = " + str(W))
	print("b = " + str(b))
	return (W, b)

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

def findNormalization(transferFunction):
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
