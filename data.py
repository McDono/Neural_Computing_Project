import numpy as np

class Dataset(object):
	def __init__(self, inputs, labels):
		self.input = inputs
		self.label = labels
		# self.inputOFF = np.array([	[], []	])
		self.inputOFF = np.array([ 	[],	[]	])
		self.inputON = np.array([ 	[],	[]	])
		# self.inputOFF[0] = np.append(self.inputOFF[0], [1], 0)
		self.inputOFF[1] = 1
		print(inputs[0][0])
		print(self.inputOFF[0])
		np.append(self.inputOFF[0], 1)
		for i in range(inputs.shape[0]):
			if labels[i] == -1 or labels[i] == 0:
				np.append(self.inputOFF[0], inputs[i][0])
				np.append(self.inputOFF[1], inputs[i][1])
			elif labels[i] == 1:
				np.append(self.inputON[0], inputs[i][0])
				np.append(self.inputON[1], inputs[i][1])

	def print_input(self):
		print(self.input)
	def print_label(self):
		print(self.label)
	def print_inputON(self):
		print(self.inputON)
	def print_inputOFF(self):
		print(self.inputOFF)


input = np.array([	[1, 1], [2, 1], [1, 2], [2, 2], [1.5, 1.5],
					[4, 4],	[5, 4],	[4, 5],	[5, 5],	[4.5, 4.5]	])
output = np.array([ -1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
data1 = Dataset(input, output)

data1.print_input()
data1.print_label()
data1.print_inputON()
data1.print_inputOFF()
