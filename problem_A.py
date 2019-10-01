
import numpy as np
import functions as fn

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


#--- DEBUG

#--- MAIN
for epoch in range(1, NOMBRE_EPOCH+1):
	print("EPOCH : #" + str(epoch))
	print("////////OFF TERMS /////////")
	for offTerm in range(0, len(inputOFF[0])):
		(W, b) = fn.run_perceptron(inputOFF, offTerm, -1, W, b, "hardlims")
	print("////////ON TERMS /////////")
	for onTerm in range(0, len(inputON[0])):
		(W, b) = fn.run_perceptron(inputON, onTerm, 1, W, b, "hardlims")
