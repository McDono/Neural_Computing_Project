import numpy as np

class Dataset(object):
	def __init__(self, inputs, labels):
		self.input = inputs
		self.label = labels
		self.inputOFF = inputs
		self.inputON = inputs
		indexON = 0
		indexOFF = 0
		for i in range(inputs.shape[0]):
			if labels[i] == -1 or labels[i] == 0:
				self.inputON = np.delete(self.inputON, i - indexON, 0)
				indexON += 1
			elif labels[i] == 1:
				self.inputOFF = np.delete(self.inputOFF, i- indexOFF, 0)
				indexOFF += 1
		self.inputOFF = self.inputOFF.T
		self.inputON = self.inputON.T
	def print_input(self):
		print(self.input)
	def print_label(self):
		print(self.label)
	def print_inputON(self):
		print(self.inputON)
	def print_inputOFF(self):
		print(self.inputOFF)

#--------- The list of all the datasets begins here -------------

input = np.array([	[1, 1], [2, 1], [1, 2], [2, 2], [1.5, 1.5],
					[4, 4],	[5, 4],	[4, 5],	[5, 5],	[4.5, 4.5]	])
output = np.array([ -1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
data1 = Dataset(input, output)
# data1.print_input()
# data1.print_label()
# data1.print_inputON()
# data1.print_inputOFF()

input = np.array([	[0.1033, 1.5372], 	[3.6839, 3.7709], 	[2.8032, 1.1594], 	[0.7604, 1.7427], 	[3.4694, 1.2937],
					[1.6739, 3.4550],	[0.9277, 3.5683],	[0.6247, 0.0667],	[2.9540, 0.2247],	[0.2690, 0.5830]	])
output = np.array([ -1, 1, 1, -1, 1, -1, -1, -1, 1, -1])
data2 = Dataset(input, output)
