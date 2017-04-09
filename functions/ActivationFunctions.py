from numpy import exp


# Sigmoid function
class Sigmoid(float or list):
	def __new__(cls, x):
		if type(x) == list:
			return [i for i in x]
		else:
			return x

	@staticmethod
	def delta(x):
		return 1
