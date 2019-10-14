from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

class ConfusionMatrix(object):
	def __init__(self, svc, dataset):
		out_pred = svc.predict(dataset.input)
		self.matrix = confusion_matrix(dataset.label, out_pred)
		# self.matrix = [[50, 10], [5, 100]]
		self.TP = self.matrix[0][0]
		self.FN = self.matrix[0][1]
		self.FP = self.matrix[1][0]
		self.TN = self.matrix[1][1]
		self.accuracy = round((self.TP + self.TN)/(self.TP+self.TN+self.FP+self.FN), 2)
		self.recall = round(self.TP/(self.TP+self.FN), 2)
		self.precision = round(self.TP/(self.TP+self.FP), 2)
		self.f_measure = round(	2*self.recall*self.precision/(self.recall+self.precision), 2)


	def print_matrix(self):
		print("Confusion matrix : ")
		print(self.matrix)
		print("TP: " + str(self.TP))
		print("FN: " + str(self.FN))
		print("FP: " + str(self.FP))
		print("TN: " + str(self.TN))
		print("Accuracy: " + str(self.accuracy))
		print("Recall: " + str(self.recall))
		print("Precision: " + str(self.precision))
		print("F-measure: " + str(self.f_measure))
