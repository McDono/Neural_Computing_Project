
import numpy as np
import functions as fn

inputOFF = np.array([[1.0, 2.0, 1.0, 2.0, 1.5],
					[1.0, 1.0, 2.0, 2.0, 1.5]])
inputON = np.array([[4.0, 4.0, 5.0, 5.0, 4.5],
					 [4.0, 5.0, 4.0, 5.0, 4.5]])

W = np.array([0.0, 0.0])
b = 0.0

NOMBRE_EPOCH = 4

#--- DEBUG

#--- MAIN
for epoch in range(1, NOMBRE_EPOCH+1):
	print("EPOCH : #" + str(epoch))
	(W, b) = fn.run_epoch(inputON, inputOFF, W, b, "hardlim")
