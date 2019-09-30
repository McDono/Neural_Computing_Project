

import numpy as np

inputOFF = np.array([[1.0, 2.0, 1.0, 2.0, 1.5],
					[1.0, 1.0, 2.0, 2.0, 1.5]])
inputON = np.array([[4.0, 4.0, 5.0, 5.0, 4.5],
					 [4.0, 5.0, 4.0, 5.0, 4.5]])
# inputTotal = np.array(inputON, inputOFF)
global W
W = np.array([0.0, 0.0])
global b
b = 0.0
# global error
# error = 0.0

NOMBRE_EPOCH = 4

def hardlims(input):
	if (input >= 0):
		return 1.0
	else:
		return -1.0

def run_neuron(input, target, W, b):
	n = np.dot(W, input) + b
	a = hardlims(n)
	error = target - a
	return error

def run_perceptron(inputTab, inputNumber, target, W, b):
	inputVector = np.array([inputTab[0][inputNumber], inputTab[1][inputNumber]])
	print("inputVector : " + str(inputVector))
	error = run_neuron(inputVector, target, W, b)
	print("error = " + str(error))
	W = W + error*0.5*inputVector.T
	b = b + error*0.5
	print("input #" + str(inputNumber))
	print("W = " + str(W))
	print("b = " + str(b))
	return (W, b)

#--- DEBUG

#--- MAIN
for epoch in range(1, NOMBRE_EPOCH+1):
	print("EPOCH : #" + str(epoch))
	print("////////OFF TERMS /////////")
	for offTerm in range(0, len(inputOFF[0])):
		(W, b) = run_perceptron(inputOFF, offTerm, -1, W, b)
	print("////////ON TERMS /////////")
	for onTerm in range(0, len(inputON[0])):
		(W, b) = run_perceptron(inputON, onTerm, 1, W, b)
