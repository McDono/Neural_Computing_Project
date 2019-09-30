inputON = [
	{1,1},
	{2,1},
	{1,2},
	{2,2},
	{1.5,1.5}
]
inputOFF = [
	{4,4},
	{4,5},
	{5,4},
	{5,5},
	{4.5,4.5}
]
W = [0, 0]
b = 0
error = 0

NOMBRE_EPOCH = 10

def hardlims(input):
	if (input >= 0):
		return 1
	else:
		return -1

def matrix_product(A,B):
	product = []
	finalComponent = []
	if (len(A[0]) != len(B)):
		return False
	else:
		for i in range(0, len(A)-1):
			finalComponent = 0
			for j in range(0, len(B)-1):
				finalComponent += A[i][j]*B[i][j]

def display_Matrix(A):
	for i in range(0, len(A)-1):
		for j in range(0, len(A[0])-1):
			print(A[i][j])

# def run_perceptron(input)
